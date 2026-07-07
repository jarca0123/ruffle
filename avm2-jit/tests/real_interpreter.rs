//! End-to-end differential test against the *real* AVM2 interpreter.
//!
//! Hand-assembles a minimal AVM2 SWF whose script-init method is pure integer
//! arithmetic (so the JIT accepts it), loads it in a real headless [`Player`]
//! with a verifying [`WasmJit`] installed, and runs a frame. The JIT executes
//! the method, then re-runs it through `run_actions` (the interpreter) and
//! compares — proving JIT == interpreter on a genuinely loaded+verified method.

#![cfg(not(target_arch = "wasm32"))]

use std::marker::PhantomData;
use std::rc::Rc;

use ruffle_avm2_jit::WasmJit;
use ruffle_core::backend::audio::NullAudioBackend;
use ruffle_core::backend::log::NullLogBackend;
use ruffle_core::backend::navigator::NullNavigatorBackend;
use ruffle_core::backend::ui::NullUiBackend;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::PlayerBuilder;

use swf::avm2::types::{
    AbcFile, ConstantPool, DefaultValue, Exception, Index, Method, MethodBody, MethodFlags,
    Multiname, Namespace, Op, Script, Trait, TraitKind,
};
use swf::avm2::write::Writer as AbcWriter;
use swf::{DoAbc2, DoAbc2Flag, FileAttributes, Header, SwfStr, Tag};

fn idx<T>(i: u32) -> Index<T> {
    Index(i, PhantomData)
}

/// Assembles an ABC with a single script-init method:
/// `getlocal0; pushscope; pushbyte 3; pushbyte 4; addi; returnvalue` → 7.
fn build_abc(ops: &[Op], max_stack: u32, num_locals: u32, doubles: Vec<f64>) -> Vec<u8> {
    let mut code = Vec::new();
    {
        let mut w = AbcWriter::new(&mut code);
        for op in ops {
            w.write_op(op).expect("write op");
        }
    }

    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![],
            uints: vec![],
            doubles,
            strings: vec![],
            namespaces: vec![],
            namespace_sets: vec![],
            multinames: vec![],
        },
        methods: vec![Method {
            name: idx(0),
            params: vec![],
            return_type: idx(0), // "*" (any)
            flags: MethodFlags::empty(),
            body: Some(idx(0)),
        }],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![],
        }],
        method_bodies: vec![MethodBody {
            method: idx(0),
            max_stack,
            num_locals,
            init_scope_depth: 0,
            max_scope_depth: 1,
            code,
            exceptions: vec![],
            traits: vec![],
        }],
    };

    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes)
        .write(abc)
        .expect("write abc");
    abc_bytes
}

/// Wraps the ABC in a minimal AVM2 SWF.
fn build_swf(abc: &[u8]) -> Vec<u8> {
    let mut header = Header::default_with_swf_version(46);
    header.num_frames = 1;
    let tags = vec![
        Tag::FileAttributes(FileAttributes::IS_ACTION_SCRIPT_3),
        Tag::DoAbc2(DoAbc2 {
            flags: DoAbc2Flag::empty(),
            name: SwfStr::from_utf8_str("test"),
            data: abc,
        }),
        Tag::ShowFrame,
    ];
    let mut out = Vec::new();
    swf::write::write_swf(&header, &tags, &mut out).expect("write swf");
    out
}

/// Builds a movie whose script-init is `ops`, loads it in a real headless
/// `Player` with a verifying JIT, runs a frame, and asserts the JIT fired and
/// agreed with the interpreter.
fn assert_jit_matches_interpreter(
    ops: &[Op],
    max_stack: u32,
    num_locals: u32,
    doubles: Vec<f64>,
) {
    let swf_bytes = build_swf(&build_abc(ops, max_stack, num_locals, doubles));
    let movie = SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None)
        .expect("valid swf");

    let jit = Rc::new(WasmJit::new().with_verify(true));

    let player = PlayerBuilder::new()
        .with_log(NullLogBackend::new())
        .with_navigator(NullNavigatorBackend::new())
        .with_audio(NullAudioBackend::new())
        .with_ui(NullUiBackend::new())
        .with_movie(movie)
        .with_autoplay(false)
        .build();

    player
        .lock()
        .unwrap()
        .set_avm2_jit_backend(jit.clone() as Rc<dyn ruffle_core::avm2::JitBackend>);

    {
        let mut p = player.lock().unwrap();
        p.preload(&mut ExecutionLimit::none());
        p.run_frame();
    }

    assert!(jit.hits() >= 1, "JIT never fired on the loaded method");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the real interpreter");
}

/// Int fast path: `getlocal0; pushscope; pushbyte 3; pushbyte 4; addi; returnvalue`.
#[test]
fn jit_matches_real_interpreter_int_path() {
    assert_jit_matches_interpreter(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::PushByte { value: 3 },
            Op::PushByte { value: 4 },
            Op::AddI,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![],
    );
}

/// Boxed (GC-aware) path: `getlocal0; pushscope; getlocal1; increment;
/// returnvalue` — a general `increment` (`ToNumber(local1)+1`; local1 is
/// `undefined` → `NaN`, so no coercion error) lowered to an imported Rust helper
/// that reaches the real `Activation`. Proves the whole helper-call chain
/// end-to-end against the interpreter.
#[test]
fn jit_matches_real_interpreter_boxed_helper() {
    assert_jit_matches_interpreter(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 1 },
            Op::Increment,
            Op::ReturnValue,
        ],
        1,
        2,
        vec![],
    );
}

/// Double fast path (inline `f64`, no helpers): `getlocal0; pushscope; pushdouble
/// 5.0; increment; returnvalue` → `6.0`. `increment` is emitted as native
/// `f64.add 1.0`, not a helper call — the whole method runs unboxed.
#[test]
fn jit_matches_real_interpreter_double_path() {
    assert_jit_matches_interpreter(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::PushDouble { value: idx(1) }, // doubles[1] = 5.0
            Op::Increment,
            Op::ReturnValue,
        ],
        1,
        1,
        vec![5.0],
    );
}

/// Double fast path with a local round-trip: `getlocal0; pushscope; pushdouble
/// 5.0; setlocal1; getlocal1; increment; returnvalue` → `6.0`. Exercises
/// `SetLocalDouble`/`GetLocalDouble` (unboxed `f64` store+load through a fresh
/// local), which the arithmetic-only case above doesn't.
#[test]
fn jit_matches_real_interpreter_double_path_setlocal() {
    assert_jit_matches_interpreter(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::PushDouble { value: idx(1) }, // doubles[1] = 5.0
            Op::SetLocal { index: 1 },
            Op::GetLocal { index: 1 },
            Op::Increment,
            Op::ReturnValue,
        ],
        2,
        2,
        vec![5.0],
    );
}

/// Assembles an ABC whose script declares a **slot trait** `g` (a `Slot` property
/// with default value `42`) on the global object, and whose script-init method is
/// `ops` (which read `g` back via `getproperty`/`getslot`). The constant pool has
/// multiname index 1 = `QName(public, "g")` for the trait name and the reads.
///
/// This exercises the GC-aware property-read JIT path end-to-end: real slot
/// resolution through `Value::get_property`/`get_slot` and a real `Value` result
/// — the same `Property::Slot` path a user class's instance slot takes, without
/// the bulk of hand-assembling `newclass`/`iinit`.
fn build_getprop_abc(ops: &[Op], max_stack: u32) -> Vec<u8> {
    let mut code = Vec::new();
    {
        let mut w = AbcWriter::new(&mut code);
        for op in ops {
            w.write_op(op).expect("write op");
        }
    }

    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![42],                            // ints[1] = 42
            uints: vec![],
            doubles: vec![],
            strings: vec![Vec::new(), b"g".to_vec()], // strings[1] = "", strings[2] = "g"
            namespaces: vec![Namespace::Package(idx(1))], // ns[1] = public ("")
            namespace_sets: vec![],
            multinames: vec![Multiname::QName {
                namespace: idx(1),
                name: idx(2),
            }], // multinames[1] = public::g
        },
        methods: vec![Method {
            name: idx(0),
            params: vec![],
            return_type: idx(0),
            flags: MethodFlags::empty(),
            body: Some(idx(0)),
        }],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![Trait {
                name: idx(1), // multinames[1] = public::g
                kind: TraitKind::Slot {
                    slot_id: 1,
                    type_name: idx(0), // "*" (untyped) — accepts the int default
                    value: Some(DefaultValue::Int(idx(1))), // 42
                },
                metadata: vec![],
                is_final: false,
                is_override: false,
            }],
        }],
        method_bodies: vec![MethodBody {
            method: idx(0),
            max_stack,
            num_locals: 1,
            init_scope_depth: 0,
            max_scope_depth: 1,
            code,
            exceptions: vec![],
            traits: vec![],
        }],
    };

    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");
    abc_bytes
}

/// Builds the slot movie from `ops`, runs a frame with a verifying JIT, and
/// asserts the JIT fired, agreed bit-for-bit with the interpreter, and produced
/// exactly int `expected`.
fn assert_getprop_reads(ops: &[Op], max_stack: u32, expected_i32: i32) {
    let swf_bytes = build_swf(&build_getprop_abc(ops, max_stack));
    let movie = SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None)
        .expect("valid swf");

    let jit = Rc::new(WasmJit::new().with_verify(true));

    let player = PlayerBuilder::new()
        .with_log(NullLogBackend::new())
        .with_navigator(NullNavigatorBackend::new())
        .with_audio(NullAudioBackend::new())
        .with_ui(NullUiBackend::new())
        .with_movie(movie)
        .with_autoplay(false)
        .build();

    player
        .lock()
        .unwrap()
        .set_avm2_jit_backend(jit.clone() as Rc<dyn ruffle_core::avm2::JitBackend>);

    {
        let mut p = player.lock().unwrap();
        p.preload(&mut ExecutionLimit::none());
        p.run_frame();
    }

    assert!(jit.hits() >= 1, "JIT never fired on the property method");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the real interpreter");

    // The method under test returned exactly `expected_i32` via the JIT. (Use the
    // result *set*, not `last_result`: many other methods also JIT per frame.)
    let expected: u64 =
        unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(expected_i32)) };
    assert!(
        jit.produced(expected),
        "no JIT'd method produced int {expected_i32} ({expected:#018x})",
    );
}

/// Typed property read → `getslot` fast path. `getlocal0; pushscope; getlocal0;
/// getproperty public::g; returnvalue`: because the receiver's type is known, the
/// verifier resolves the read to `GetSlot { index: 0 }` — a direct slot fetch
/// (one of the hottest AVM2 ops), JIT-compiled via the arity-2 `gs` helper.
#[test]
fn jit_matches_real_interpreter_getslot() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::GetProperty { index: idx(1) }, // multinames[1] = public::g
            Op::ReturnValue,
        ],
        2,
        42,
    );
}

/// Dynamic property read → real `getproperty` (multiname) path. Inserting
/// `coerce_a` erases the receiver's static type to `*`, so the verifier can *not*
/// resolve a slot and keeps `GetPropertyStatic { multiname }`. The JIT compiles it
/// via the arity-2 `gp` helper + the per-run multiname table, resolving `public::g`
/// through `Value::get_property` at run time — still reaching the same slot (`42`).
#[test]
fn jit_matches_real_interpreter_getproperty_dynamic() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::CoerceA, // erase type to `*` → defeats slot specialization
            Op::GetProperty { index: idx(1) },
            Op::ReturnValue,
        ],
        2,
        42,
    );
}

/// Slot **write** path → `setslot` + boxed const. Writes `99` into slot `g`, then
/// reads it back and returns it: `getlocal0; pushscope; getlocal0; pushbyte 99;
/// setproperty public::g; getlocal0; getproperty public::g; returnvalue` → `99`.
/// The verifier resolves the typed write to a `SetSlot`/`SetSlotNoCoerce`, which
/// the JIT compiles via the arity-3 set helper (`PushIntValue` supplies the 99).
/// Overwrites the slot's `42` default, so a result of `99` proves the write landed.
#[test]
fn jit_matches_real_interpreter_setslot() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },        // receiver for the write
            Op::PushByte { value: 99 },       // value
            Op::SetProperty { index: idx(1) }, // g = 99  → verifier: setslot
            Op::GetLocal { index: 0 },        // receiver for the read-back
            Op::GetProperty { index: idx(1) }, // read g → getslot
            Op::ReturnValue,
        ],
        2,
        99,
    );
}

/// Generic (untyped) `multiply` → arity-2 two-stack helper. Reads `g` (=42) with
/// `coerce_a` so it's a `*`-typed `Integer` (defeating the int/double fast paths),
/// then `* 2`. Both operands unpack to `Integer`, so the helper's `int*int→int`
/// fast path fires → `Integer(84)` — the *same* bits the interpreter produces
/// (a `Number` result would diverge). `getlocal0; pushscope; getlocal0; coerce_a;
/// getproperty g; pushbyte 2; multiply; returnvalue` → `84`.
#[test]
fn jit_matches_real_interpreter_multiply() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::CoerceA,
            Op::GetProperty { index: idx(1) }, // g = 42, typed `*`
            Op::PushByte { value: 2 },
            Op::Multiply, // 42 * 2 → Integer(84) via the int fast path
            Op::ReturnValue,
        ],
        2,
        84,
    );
}

/// Generic `subtract` → arity-2 helper, int fast path (`int-int→int` via
/// `checked_sub`): `g(42) - 2` → `Integer(40)`, bit-matching the interpreter.
#[test]
fn jit_matches_real_interpreter_subtract() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::CoerceA,
            Op::GetProperty { index: idx(1) },
            Op::PushByte { value: 2 },
            Op::Subtract, // 42 - 2 → Integer(40)
            Op::ReturnValue,
        ],
        2,
        40,
    );
}

/// Generic `divide` (always a `Number` result) chained into `coerce_i`
/// (`Number`→`Integer`): `ToInt32(g(42) / 4)` = `ToInt32(10.5)` = `10`. Exercises
/// the `divide` arity-2 helper *and* the `coerce_i` arity-1 helper, verified
/// bit-for-bit against the interpreter.
#[test]
fn jit_matches_real_interpreter_divide_coerce_i() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::CoerceA,
            Op::GetProperty { index: idx(1) },
            Op::PushByte { value: 4 },
            Op::Divide,  // 42 / 4 → Number(10.5)
            Op::CoerceI, // ToInt32(10.5) → Integer(10)
            Op::ReturnValue,
        ],
        2,
        10,
    );
}

/// Boxed comparison → `equals` (arity-2 two-stack helper). `getlocal0; pushscope;
/// getlocal0; getlocal0; equals; returnvalue`: `this == this` → `true` (a `Boolean`
/// `Value`), through `abstract_eq`, verified against the interpreter.
#[test]
fn jit_matches_real_interpreter_equals() {
    let ops = [
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 0 },
        Op::GetLocal { index: 0 },
        Op::Equals, // this == this → true
        Op::ReturnValue,
    ];
    let swf_bytes = build_swf(&build_abc(&ops, 2, 1, vec![]));
    let movie = SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None)
        .expect("valid swf");

    let jit = Rc::new(WasmJit::new().with_verify(true));
    let player = PlayerBuilder::new()
        .with_log(NullLogBackend::new())
        .with_navigator(NullNavigatorBackend::new())
        .with_audio(NullAudioBackend::new())
        .with_ui(NullUiBackend::new())
        .with_movie(movie)
        .with_autoplay(false)
        .build();
    player
        .lock()
        .unwrap()
        .set_avm2_jit_backend(jit.clone() as Rc<dyn ruffle_core::avm2::JitBackend>);
    {
        let mut p = player.lock().unwrap();
        p.preload(&mut ExecutionLimit::none());
        p.run_frame();
    }

    assert!(jit.hits() >= 1, "JIT never fired on the equals method");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the real interpreter");
    let expected: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(true)) };
    assert!(jit.produced(expected), "equals did not yield true");
}

/// domainMemory round-trip → `si32`/`li32` (the FlasCC "RAM" ops). `getlocal0;
/// pushscope; pushbyte 123; pushbyte 0; si32; pushbyte 0; li32; returnvalue`:
/// stores `123` at `domainMemory[0]` then loads it back → `123`. The default
/// domain memory is 1024 zeroed bytes, so address 0 is in bounds. Verified against
/// the interpreter (production runs with verify off, so this guards the store/load
/// helpers).
#[test]
fn jit_matches_real_interpreter_domain_memory() {
    let ops = [
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::PushByte { value: 123 }, // value
        Op::PushByte { value: 0 },   // address
        Op::Si32,                    // domainMemory[0] = 123
        Op::PushByte { value: 0 },   // address
        Op::Li32,                    // load domainMemory[0]
        Op::ReturnValue,
    ];
    let swf_bytes = build_swf(&build_abc(&ops, 2, 1, vec![]));
    let movie = SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None)
        .expect("valid swf");

    let jit = Rc::new(WasmJit::new().with_verify(true));
    let player = PlayerBuilder::new()
        .with_log(NullLogBackend::new())
        .with_navigator(NullNavigatorBackend::new())
        .with_audio(NullAudioBackend::new())
        .with_ui(NullUiBackend::new())
        .with_movie(movie)
        .with_autoplay(false)
        .build();
    player
        .lock()
        .unwrap()
        .set_avm2_jit_backend(jit.clone() as Rc<dyn ruffle_core::avm2::JitBackend>);
    {
        let mut p = player.lock().unwrap();
        p.preload(&mut ExecutionLimit::none());
        p.run_frame();
    }

    assert!(jit.hits() >= 1, "JIT never fired on the domainMemory method");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the real interpreter");
    let expected: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(123i32)) };
    assert!(jit.produced(expected), "domainMemory round-trip did not yield 123");
}

/// Real scope access → `getscopeobject`. `getlocal0; pushscope; getscopeobject 0;
/// getproperty public::g; returnvalue`: `pushscope` pushes `this` (the global)
/// onto the real Activation scope stack, `getscopeobject 0` reads it back, and the
/// property read pulls slot `g` off it → `42`. Exercises the `push_scope` /
/// `get_scope_object` helpers *and* the verifier's scope reset (the JIT's real
/// scope push must not corrupt the interpreter re-run).
#[test]
fn jit_matches_real_interpreter_getscopeobject() {
    assert_getprop_reads(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetScopeObject { index: 0 }, // the pushed `this`, via the scope stack
            Op::GetProperty { index: idx(1) }, // read g off it → 42
            Op::ReturnValue,
        ],
        2,
        42,
    );
}

/// `istypelate` with a non-class type operand must **throw `#1041`** (matching the
/// interpreter's `op_is_type_late`), not swallow to `false` — and the thrown error
/// must dispatch through the method's exception handlers. The method is
/// `try { this is local1 /* undefined → #1041 */ ; return } catch { return 77 }`:
/// with the throw wired the JIT lands in the catch block and returns `77`; the old
/// swallowing behavior returned `false` and never reached the handler (observed in
/// the wild as runaway recursion/allocation when a game's control flow relied on
/// catching `#1041`).
#[test]
fn jit_matches_real_interpreter_istypelate_throws_1041() {
    let ops = [
        Op::GetLocal { index: 0 },  // byte 0
        Op::PushScope,              // byte 1
        Op::GetLocal { index: 0 },  // byte 2   ─┐ try range
        Op::GetLocal { index: 1 },  // byte 3    │ (local1 = undefined)
        Op::IsTypeLate,             // byte 4    │ `this is undefined` → throws #1041
        Op::ReturnValue,            // byte 5   ─┘ (unreachable)
        Op::NewCatch { index: idx(0) }, // byte 6: catch target (stack: [exc, scope])
        Op::Pop,                    // byte 8
        Op::Pop,                    // byte 9
        Op::PushByte { value: 77 }, // byte 10
        Op::ReturnValue,            // byte 12
    ];
    let mut code = Vec::new();
    {
        let mut w = AbcWriter::new(&mut code);
        for op in &ops {
            w.write_op(op).expect("write op");
        }
    }
    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![],
            uints: vec![],
            doubles: vec![],
            strings: vec![],
            namespaces: vec![],
            namespace_sets: vec![],
            multinames: vec![],
        },
        methods: vec![Method {
            name: idx(0),
            params: vec![],
            return_type: idx(0), // "*" (any)
            flags: MethodFlags::empty(),
            body: Some(idx(0)),
        }],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![],
        }],
        method_bodies: vec![MethodBody {
            method: idx(0),
            max_stack: 2,
            num_locals: 2,
            init_scope_depth: 0,
            max_scope_depth: 2, // body pushscope + the catch scope
            code,
            exceptions: vec![Exception {
                from_offset: 2,
                to_offset: 5,
                target_offset: 6,
                variable_name: idx(0), // unnamed
                type_name: idx(0),     // catch-all ("*")
            }],
            traits: vec![],
        }],
    };
    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");

    let swf_bytes = build_swf(&abc_bytes);
    let movie = SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None)
        .expect("valid swf");

    let jit = Rc::new(WasmJit::new().with_verify(true));
    let player = PlayerBuilder::new()
        .with_log(NullLogBackend::new())
        .with_navigator(NullNavigatorBackend::new())
        .with_audio(NullAudioBackend::new())
        .with_ui(NullUiBackend::new())
        .with_movie(movie)
        .with_autoplay(false)
        .build();
    player
        .lock()
        .unwrap()
        .set_avm2_jit_backend(jit.clone() as Rc<dyn ruffle_core::avm2::JitBackend>);
    {
        let mut p = player.lock().unwrap();
        p.preload(&mut ExecutionLimit::none());
        p.run_frame();
    }

    assert!(jit.hits() >= 1, "JIT never fired on the istypelate method");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the real interpreter");
    let expected: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(77i32)) };
    assert!(
        jit.produced(expected),
        "istypelate did not throw #1041 into the catch handler (77 never returned)"
    );
    let swallowed: u64 = unsafe { std::mem::transmute(ruffle_core::avm2::Value::from(false)) };
    assert!(
        !jit.produced(swallowed),
        "istypelate swallowed the non-class type to `false` instead of throwing"
    );
}

// NOTE: Boxed **control flow** (branches/loops) is validated at the JitOp level by
// `lower::tests::lowers_boxed_countdown_loop`, which drives the exact `compile()`
// dispatch-loop path a real method takes (boxed `IfFalseBoxed` condition via the
// `to_boolean` helper + a `Jump` back-edge). A loaded-SWF loop is intentionally
// not tested here: AVM2 branch offsets are byte-relative and hand-assembling a
// loop that the verifier accepts is brittle (same reason the int-path loops are
// covered only at the JitOp level).
