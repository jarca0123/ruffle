//! WASM-emitting JIT backend for AVM2 (`ruffle_avm2_jit`).
//!
//! Implements [`ruffle_core::avm2::JitBackend`] by compiling hot AVM2 methods to
//! a WebAssembly module at runtime and running it:
//!
//! - **Web**: emit WASM bytes, hand them to the browser's engine
//!   (`WebAssembly.Module` / `Instance` via a JS host import), and call the
//!   compiled function through a shared table. The generated module imports
//!   Ruffle's linear memory + a table of Rust "runtime" helpers.
//! - **Native** (tests, and a future desktop path): the same emitted module can
//!   be validated / executed through a WASM runtime, so JIT↔interpreter
//!   equivalence is testable without a browser. (The desktop production path
//!   would instead use a native code generator such as cranelift.)
//!
//! ## Execution model (see [`emit`])
//! The whole AVM2 method state — registers `[0..num_locals]` and the operand
//! stack above them — is one contiguous run of 8-byte NaN-boxed `Value` slots
//! (`ruffle_core`'s stack). The JIT receives the frame's base pointer
//! (`state_ptr`) and addresses slot `i` at `state_ptr + i*8`. The operand stack
//! is simulated at *compile* time, so straight-line code uses fixed offsets and
//! needs no runtime stack pointer. Anything GC-aware (property access, calls,
//! allocation, throwing coercions) is emitted as a `call` to an imported Rust
//! helper, keeping GC correctness in Rust.
//!
//! Status: prototype. [`WasmJit::try_run`] compiles the supported numeric +
//! control-flow subset (via [`translate`] → [`lower::compile`]) and runs it
//! natively through [`runner`]; everything else declines to the interpreter. The
//! web execution path (browser bridge over shared memory) is not wired up yet,
//! so on `wasm32` it also declines.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use ruffle_core::avm2::{Activation, Error, JitBackend, Method, Value};

pub mod emit;
pub mod lower;
pub mod runner;
pub mod translate;

/// Per-method compilation outcome held in the cache.
/// `None` marks a method as *known-unsupported* so it is never re-translated.
type CacheEntry = Option<Rc<[u8]>>;

/// A [`JitBackend`] that compiles AVM2 methods by emitting WebAssembly at runtime.
///
/// Install with `avm2.set_jit_backend(WasmJit::shared())`.
pub struct WasmJit {
    /// Compiled module bytes keyed by [`Method::as_ptr`]. `Some(None)` records a
    /// method we've already found unsupported so we don't retranslate it.
    cache: RefCell<HashMap<usize, CacheEntry>>,
    /// When set, every JIT run is also executed through the real interpreter and
    /// the two results are asserted equal (differential self-check). Opt-in —
    /// only sound for the side-effect-free methods the JIT accepts.
    verify: bool,
    /// Re-entrancy guard: while the verifier is running the interpreter, nested
    /// [`Self::try_run`] calls decline so the interpreter actually interprets.
    in_verify: Cell<bool>,
    /// Number of methods actually executed by the JIT.
    hits: Cell<u32>,
    /// Number of JIT/interpreter divergences seen under `verify` (should be 0).
    mismatches: Cell<u32>,
}

impl Default for WasmJit {
    fn default() -> Self {
        Self {
            cache: RefCell::new(HashMap::new()),
            verify: false,
            in_verify: Cell::new(false),
            hits: Cell::new(0),
            mismatches: Cell::new(0),
        }
    }
}

impl WasmJit {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables the differential self-check (compare every JIT run against the
    /// interpreter). For testing/validation; do not enable in production.
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }

    /// Boxed as `Rc<dyn JitBackend>` for
    /// [`Avm2::set_jit_backend`](ruffle_core::avm2::Avm2::set_jit_backend).
    pub fn shared() -> Rc<dyn JitBackend> {
        Rc::new(Self::new())
    }

    /// Number of methods the JIT has executed.
    pub fn hits(&self) -> u32 {
        self.hits.get()
    }

    /// Number of JIT/interpreter divergences seen under [`Self::with_verify`].
    pub fn mismatches(&self) -> u32 {
        self.mismatches.get()
    }

    /// Returns the compiled module for `method`, compiling+caching on first use.
    /// `None` means unsupported (cached so we don't retry).
    fn compiled(&self, method: Method<'_>) -> Option<Rc<[u8]>> {
        let key = method.as_ptr() as usize;
        if let Some(entry) = self.cache.borrow().get(&key) {
            return entry.clone();
        }
        // Compile outside the borrow; the parsed ops are read-only.
        let entry: CacheEntry = (|| {
            method.body()?;
            let ops = translate::translate(&method.get_verified_info().parsed_code)?;
            Some(Rc::from(lower::compile(&ops)?.into_boxed_slice()))
        })();
        self.cache.borrow_mut().insert(key, entry.clone());
        entry
    }
}

/// Reinterprets a `Value` as its 8-byte NaN-boxed bit pattern.
fn value_to_bits(v: Value<'_>) -> u64 {
    debug_assert_eq!(std::mem::size_of::<Value<'_>>(), 8);
    // SAFETY: `Value` is a NaN-boxed `u64`; transmuting to its bits is total.
    unsafe { std::mem::transmute(v) }
}

/// Reconstructs a `Value` from bits produced by the JIT.
///
/// SAFETY / soundness: only sound when `bits` encodes a non-pointer `Value`
/// (int/number/bool/etc.) — fabricating a boxed `Gc` pointer this way would skip
/// the write barrier and be unsound. The prototype only compiles `int`-typed
/// numeric methods, whose results are always `int` values, so this holds. See
/// [`translate`]'s soundness note.
unsafe fn value_from_bits<'gc>(bits: u64) -> Value<'gc> {
    // SAFETY: delegated to the caller (see doc comment).
    unsafe { std::mem::transmute(bits) }
}

impl JitBackend for WasmJit {
    fn try_run<'gc>(
        &self,
        activation: &mut Activation<'_, 'gc>,
        method: Method<'gc>,
    ) -> Option<Result<Value<'gc>, Error<'gc>>> {
        // Decline while the verifier is driving the interpreter (avoid recursion).
        if self.in_verify.get() {
            return None;
        }

        let num_locals = method.body()?.num_locals as usize;
        let bytes = self.compiled(method)?;

        // Snapshot the frame's registers as `Value` bits *before* running
        // anything, so the interpreter self-check sees a pristine frame. The
        // operand stack is empty at entry; SetLocal only writes indices <
        // num_locals.
        let mut regs = Vec::with_capacity(num_locals);
        for i in 0..num_locals as u32 {
            regs.push(value_to_bits(activation.local_register(i)));
        }

        let result_bits = runner::run(&bytes, &regs)?;
        // SAFETY: supported methods return `int` values (non-pointer). See above.
        let jit_value = unsafe { value_from_bits(result_bits) };
        self.hits.set(self.hits.get() + 1);

        if self.verify {
            // Run the real interpreter on the same (still-pristine) frame and
            // compare. The guard makes the nested call interpret. Divergences are
            // recorded (not panicked) so callers can inspect them even if the
            // host swallows panics.
            self.in_verify.set(true);
            let interp = activation.run_actions(method);
            self.in_verify.set(false);
            let agrees = matches!(interp, Ok(v) if value_to_bits(v) == result_bits);
            if !agrees {
                self.mismatches.set(self.mismatches.get() + 1);
                tracing::error!(
                    "JIT/interpreter divergence in method {:p}: jit={result_bits:#018x}",
                    method.as_ptr()
                );
            }
        }

        Some(Ok(jit_value))
    }
}
