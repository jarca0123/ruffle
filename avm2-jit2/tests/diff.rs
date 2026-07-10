//! End-to-end differential test against the *real* AVM2 interpreter.
//!
//! Hand-assembles a minimal AVM2 SWF whose script-init method is in the step-1
//! numeric subset, loads it in a real headless [`Player`] with a verifying
//! [`Jit2`] installed, and runs a frame. In verify mode the JIT executes the
//! method, then re-runs it through `run_actions` (the interpreter) and compares —
//! proving JIT == interpreter, bit-for-bit, on a genuinely loaded+verified method.
//!
//! The script-init methods here deliberately omit the usual `getlocal0; pushscope`
//! prologue (the step-1 subset doesn't cover scope ops yet), so they are pure
//! numeric straight-line code.

#![cfg(not(target_arch = "wasm32"))]

use std::marker::PhantomData;
use std::rc::Rc;

use ruffle_avm2_jit2::Jit2;
use ruffle_core::backend::audio::NullAudioBackend;
use ruffle_core::backend::log::NullLogBackend;
use ruffle_core::backend::navigator::NullNavigatorBackend;
use ruffle_core::backend::ui::NullUiBackend;
use ruffle_core::limits::ExecutionLimit;
use ruffle_core::tag_utils::SwfMovie;
use ruffle_core::PlayerBuilder;

use swf::avm2::types::{
    AbcFile, ConstantPool, DefaultValue, Index, Method, MethodBody, MethodFlags, MethodParam,
    Multiname, Namespace, Op, Script, Trait, TraitKind,
};
use swf::avm2::write::Writer as AbcWriter;
use swf::{DoAbc2, DoAbc2Flag, FileAttributes, Header, SwfStr, Tag};

fn idx<T>(i: u32) -> Index<T> {
    Index(i, PhantomData)
}

/// Branch target of a control-flow op (whose `offset` field this test treats as
/// an **op index**, not the ABC byte offset).
fn branch_target(op: &Op) -> Option<usize> {
    match *op {
        Op::Jump { offset }
        | Op::IfTrue { offset }
        | Op::IfFalse { offset }
        | Op::IfGe { offset }
        | Op::IfGt { offset }
        | Op::IfLe { offset }
        | Op::IfLt { offset } => Some(offset as usize),
        _ => None,
    }
}

/// Return `op` with its branch offset replaced.
fn with_offset(op: &Op, off: i32) -> Op {
    match *op {
        Op::Jump { .. } => Op::Jump { offset: off },
        Op::IfTrue { .. } => Op::IfTrue { offset: off },
        Op::IfFalse { .. } => Op::IfFalse { offset: off },
        Op::IfGe { .. } => Op::IfGe { offset: off },
        Op::IfGt { .. } => Op::IfGt { offset: off },
        Op::IfLe { .. } => Op::IfLe { offset: off },
        Op::IfLt { .. } => Op::IfLt { offset: off },
        _ => unreachable!("with_offset on a non-branch op"),
    }
}

/// Assemble ops whose branch offsets are **op indices** into ABC bytecode,
/// resolving each to the byte offset the format needs (relative to the byte after
/// the branch instruction). Two passes: measure each op's encoded size (a
/// branch is always opcode + i24, independent of the offset value), then rewrite
/// branches with the correct byte offsets.
fn assemble(ops: &[Op]) -> Vec<u8> {
    let mut starts = vec![0usize];
    for op in ops {
        let mut one = Vec::new();
        AbcWriter::new(&mut one).write_op(op).expect("measure op");
        starts.push(starts.last().unwrap() + one.len());
    }

    let resolved: Vec<Op> = ops
        .iter()
        .enumerate()
        .map(|(i, op)| match branch_target(op) {
            Some(target) => {
                let byte_off = starts[target] as i32 - starts[i + 1] as i32;
                with_offset(op, byte_off)
            }
            None => op.clone(),
        })
        .collect();

    let mut code = Vec::new();
    {
        let mut w = AbcWriter::new(&mut code);
        for op in &resolved {
            w.write_op(op).expect("write op");
        }
    }
    code
}

/// Assemble an ABC whose single script-init method is `ops`.
fn build_abc(
    ops: &[Op],
    max_stack: u32,
    num_locals: u32,
    ints: Vec<i32>,
    doubles: Vec<f64>,
) -> Vec<u8> {
    // Branch offsets in `ops` are op indices; `assemble` resolves them to byte
    // offsets. For branch-free methods it's a straight re-encode.
    let code = assemble(ops);

    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints,
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
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");
    abc_bytes
}

/// Wrap the ABC in a minimal AVM2 SWF.
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

/// Build a movie whose script-init is `ops`, load it in a real headless `Player`
/// with a verifying JIT, run a frame, and assert the JIT fired and agreed with
/// the interpreter.
fn assert_jit_matches_interpreter(
    ops: &[Op],
    max_stack: u32,
    num_locals: u32,
    ints: Vec<i32>,
    doubles: Vec<f64>,
) {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::WARN)
        .try_init();

    let swf_bytes = build_swf(&build_abc(ops, max_stack, num_locals, ints, doubles));
    let movie =
        SwfMovie::from_data(&swf_bytes, "test.swf".to_string(), false, None).expect("valid swf");

    let jit = Rc::new(Jit2::new().with_verify(true));

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

/// `pushbyte 3; pushbyte 4; addi; returnvalue` → Integer(7).
#[test]
fn addi_int_path() {
    assert_jit_matches_interpreter(
        &[
            Op::PushByte { value: 3 },
            Op::PushByte { value: 4 },
            Op::AddI,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![],
    );
}

/// The real prologue: `getlocal0; pushscope; pushbyte 3; pushbyte 4; addi;
/// returnvalue` → Integer(7). Proves methods carrying the standard scope prologue
/// are JIT'd and still match the interpreter.
#[test]
fn scope_prologue() {
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
        vec![],
    );
}

/// `pushbyte 6; pushbyte 7; multiply; returnvalue` → Integer(42) (in-range, so the
/// overflow-capable multiply keeps the `Integer` tag).
#[test]
fn multiply_ints_in_range() {
    assert_jit_matches_interpreter(
        &[
            Op::PushByte { value: 6 },
            Op::PushByte { value: 7 },
            Op::Multiply,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![],
    );
}

/// `pushint 50000; pushint 50000; multiply; returnvalue` → Number(2.5e9): overflow
/// flips the tag to `Number`, and the JIT must match that exactly.
#[test]
fn multiply_ints_overflow_to_number() {
    assert_jit_matches_interpreter(
        &[
            Op::PushInt { value: idx(1) }, // ints[1] = 50000
            Op::PushInt { value: idx(1) },
            Op::Multiply,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![50_000],
        vec![],
    );
}

/// `pushdouble 1.5; pushdouble 2.0; add; returnvalue` → Number(3.5).
#[test]
fn number_add() {
    assert_jit_matches_interpreter(
        &[
            Op::PushDouble { value: idx(1) }, // doubles[1] = 1.5
            Op::PushDouble { value: idx(2) }, // doubles[2] = 2.0
            Op::Add,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![1.5, 2.0],
    );
}

/// A real loop: `sum = 0; for (i = 0; i < 10; i++) sum += i; return sum` → 45.
/// The counter/accumulator are locals pinned in the entry block; the loop runs
/// through the `br_table` dispatch and must agree with the interpreter exactly.
#[test]
fn sum_loop() {
    use Op::*;
    // locals: 0 = this (unused), 1 = sum, 2 = i. Branch offsets are OP INDICES
    // (assemble resolves them). Standalone `lessthan` + `iffalse` (1:1 with core
    // ops; the combined `ifge` would be split). The loop header carries a `Label`
    // op because the verifier requires a backward jump to target one (else #1021);
    // it folds to Nop in the verified code.
    assert_jit_matches_interpreter(
        &[
            PushByte { value: 0 },       // 0
            SetLocal { index: 1 },       // 1  sum = 0
            PushByte { value: 0 },       // 2
            SetLocal { index: 2 },       // 3  i = 0
            Label,                       // 4  header (backward-jump target)
            GetLocal { index: 2 },       // 5
            PushByte { value: 10 },      // 6
            LessThan,                    // 7  i < 10 ?
            IfFalse { offset: 15 },      // 8  if !(i < 10) goto exit (op 15)
            GetLocal { index: 1 },       // 9  body
            GetLocal { index: 2 },       // 10
            AddI,                        // 11
            SetLocal { index: 1 },       // 12 sum += i
            IncLocalI { index: 2 },      // 13 i++
            Jump { offset: 4 },          // 14 back to header (the Label)
            GetLocal { index: 1 },       // 15 exit
            ReturnValue,                 // 16
        ],
        2,
        3,
        vec![],
        vec![],
    );
}

/// A ternary whose result crosses the merge on the operand stack (non-empty
/// boundary, step 2): `3 > 4 ? 10 : 20` → 20, matching the interpreter.
#[test]
fn ternary_operand_crosses_block() {
    use Op::*;
    assert_jit_matches_interpreter(
        &[
            PushByte { value: 3 },  // 0
            PushByte { value: 4 },  // 1
            GreaterThan,            // 2
            IfFalse { offset: 6 },  // 3  if !(3>4) goto op 6
            PushByte { value: 10 }, // 4
            Jump { offset: 7 },     // 5  goto op 7
            PushByte { value: 20 }, // 6
            ReturnValue,            // 7  merge: value (10 or 20) on stack
        ],
        2,
        1,
        vec![],
        vec![],
    );
}

/// Early-bound slot read (step 3b). The script declares a global slot `g` = 42.
/// `getlocal0; pushscope; getlocal0; getproperty public::g; returnvalue`: the
/// verifier, knowing the receiver's class, resolves the read to `GetSlot{index:0}`
/// — which the JIT compiles via the `get_slot` helper import. Result 42, matching
/// the interpreter.
#[test]
fn getslot_reads_global() {
    // The script-init has one global slot trait `g` (slot_id 1, default int 42).
    // `getproperty g` on the typed `this` resolves to GetSlot{0}.
    run_and_check(&build_swf(&build_slot_abc(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::GetProperty { index: idx(1) },
            Op::ReturnValue,
        ],
        2,
    )));
}

/// Early-bound slot WRITE (step 3b): `getlocal0; pushscope; getlocal0; pushbyte 99;
/// setproperty g; getlocal0; getproperty g; returnvalue`. The verifier resolves
/// `setproperty g` to `SetSlotNoCoerce{0}` (via the `set_slot` helper) and
/// `getproperty g` to `GetSlot{0}`; the read-back returns 99, matching the
/// interpreter (proving the write happened through the helper + GC barrier).
#[test]
fn setslot_then_getslot() {
    run_and_check(&build_swf(&build_slot_abc(
        &[
            Op::GetLocal { index: 0 },
            Op::PushScope,
            Op::GetLocal { index: 0 },
            Op::PushByte { value: 99 },
            Op::SetProperty { index: idx(1) }, // → SetSlotNoCoerce{0}
            Op::GetLocal { index: 0 },
            Op::GetProperty { index: idx(1) }, // → GetSlot{0}
            Op::ReturnValue,
        ],
        3,
    )));
}

/// JIT→JIT call (step 4): the script defines a method `f()` returning 42; the
/// init `getlocal0; pushscope; getlocal0; callproperty f, 0; returnvalue` — the
/// verifier resolves the typed call to `CallMethod`, which the JIT dispatches via
/// the `call_method` helper. Result 42, matching the interpreter.
#[test]
fn calls_a_method() {
    let init_ops = assemble(&[
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 0 },
        Op::CallProperty {
            index: idx(1),
            num_args: 0,
        }, // → CallMethod
        Op::ReturnValue,
    ]);
    let f_ops = assemble(&[Op::PushByte { value: 42 }, Op::ReturnValue]);

    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![],
            uints: vec![],
            doubles: vec![],
            strings: vec![b"".to_vec(), b"f".to_vec()],
            namespaces: vec![Namespace::Package(idx(1))],
            multinames: vec![Multiname::QName {
                namespace: idx(1),
                name: idx(2),
            }], // multinames[1] = public::f
            namespace_sets: vec![],
        },
        methods: vec![
            Method {
                name: idx(0),
                params: vec![],
                return_type: idx(0),
                flags: MethodFlags::empty(),
                body: Some(idx(0)),
            },
            Method {
                name: idx(0),
                params: vec![],
                return_type: idx(0),
                flags: MethodFlags::empty(),
                body: Some(idx(1)),
            },
        ],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![Trait {
                name: idx(1), // public::f
                kind: TraitKind::Method {
                    disp_id: 1,
                    method: idx(1),
                },
                metadata: vec![],
                is_final: false,
                is_override: false,
            }],
        }],
        method_bodies: vec![
            MethodBody {
                method: idx(0),
                max_stack: 2,
                num_locals: 1,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: init_ops,
                exceptions: vec![],
                traits: vec![],
            },
            MethodBody {
                method: idx(1),
                max_stack: 1,
                num_locals: 1,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: f_ops,
                exceptions: vec![],
                traits: vec![],
            },
        ],
    };
    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");
    run_and_check(&build_swf(&abc_bytes));
}

/// Direct self-recursive call (step 4, direct JIT→JIT `call`): a method
/// `rec(n:int, acc:int)` that returns `acc` when `n < 1` and otherwise tail-calls
/// `rec(n-1, acc+n)` on `this`. The verifier lowers the self `callproperty` to a
/// `CallMethod` whose disp_id resolves (against `this`'s vtable) back to `rec`, so
/// the JIT emits a direct WASM `call` to `rec`'s own `run` — no interpreter
/// trampoline. `rec(5, 0)` accumulates 5+4+3+2+1 = 15; the JIT result must match
/// the interpreter bit-for-bit, and the direct path must actually fire.
#[test]
fn self_recursive_call_is_direct() {
    // init: rec(5, 0). `getlocal0; pushscope; getlocal0; pushbyte 5; pushbyte 0;
    // callproperty rec,2; returnvalue`.
    let init_ops = assemble(&[
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 0 },
        Op::PushByte { value: 5 },
        Op::PushByte { value: 0 },
        Op::CallProperty {
            index: idx(1),
            num_args: 2,
        },
        Op::ReturnValue,
    ]);
    // rec body (branch offsets are OP INDICES; `assemble` resolves them). No
    // back-edge (recursion is a call, not a jump), so no `Label` is needed.
    let rec_ops = assemble(&[
        Op::GetLocal { index: 0 },   // 0  this (scope)
        Op::PushScope,               // 1
        Op::GetLocal { index: 1 },   // 2  n
        Op::PushByte { value: 1 },   // 3
        Op::LessThan,                // 4  n < 1  ⇔ n <= 0
        Op::IfTrue { offset: 15 },   // 5  → base
        Op::GetLocal { index: 0 },   // 6  this (receiver)
        Op::GetLocal { index: 1 },   // 7  n
        Op::PushByte { value: 1 },   // 8
        Op::SubtractI,               // 9  n-1 (int)
        Op::GetLocal { index: 2 },   // 10 acc
        Op::GetLocal { index: 1 },   // 11 n
        Op::AddI,                    // 12 acc+n (int)
        Op::CallProperty {
            index: idx(1),
            num_args: 2,
        }, // 13 rec(n-1, acc+n) → CallMethod (self)
        Op::ReturnValue,             // 14 return the recursive result
        Op::GetLocal { index: 2 },   // 15 base: acc
        Op::ReturnValue,             // 16
    ]);

    let int_param = || MethodParam {
        name: None,
        kind: idx(2), // multinames[2] = public::int
        default_value: None,
    };
    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![],
            uints: vec![],
            doubles: vec![],
            strings: vec![b"".to_vec(), b"rec".to_vec(), b"int".to_vec()],
            namespaces: vec![Namespace::Package(idx(1))], // ns[1] = public ("")
            namespace_sets: vec![],
            multinames: vec![
                Multiname::QName {
                    namespace: idx(1),
                    name: idx(2),
                }, // [1] = public::rec
                Multiname::QName {
                    namespace: idx(1),
                    name: idx(3),
                }, // [2] = public::int
            ],
        },
        methods: vec![
            Method {
                name: idx(0),
                params: vec![],
                return_type: idx(0),
                flags: MethodFlags::empty(),
                body: Some(idx(0)),
            },
            Method {
                name: idx(0),
                params: vec![int_param(), int_param()], // rec(n:int, acc:int)
                return_type: idx(0),
                flags: MethodFlags::empty(),
                body: Some(idx(1)),
            },
        ],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![Trait {
                name: idx(1), // public::rec
                kind: TraitKind::Method {
                    disp_id: 1,
                    method: idx(1),
                },
                metadata: vec![],
                is_final: false,
                is_override: false,
            }],
        }],
        method_bodies: vec![
            MethodBody {
                method: idx(0),
                max_stack: 3,
                num_locals: 1,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: init_ops,
                exceptions: vec![],
                traits: vec![],
            },
            MethodBody {
                method: idx(1),
                max_stack: 4,
                num_locals: 3,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: rec_ops,
                exceptions: vec![],
                traits: vec![],
            },
        ],
    };
    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");

    let jit = run_and_get_jit(&build_swf(&abc_bytes));
    assert!(
        jit.direct_calls() >= 1,
        "the direct self-call path never fired (rec fell back to the trampoline)"
    );
}

/// General direct JIT→JIT call (step 4 amalgam): a `caller(n:int)` that returns
/// `this.helper(n)`, where `helper(n:int)` returns `n + 100`. `helper` is a
/// *different* method, so this exercises the per-caller call-tree amalgam:
/// `caller`'s module co-compiles `helper` and calls it with a direct WASM `call`
/// (native arg), not the trampoline. The script-init calls `helper` first (so it
/// is verified before `caller` compiles), then `caller(7)` → `helper(7)` = 107.
/// The JIT result must match the interpreter, and the direct path must fire.
#[test]
fn sibling_call_is_direct() {
    // init: helper(7) [discard], then caller(7). Calling helper first ensures it
    // is verified by the time `caller` is JIT-compiled (so it can be co-compiled).
    let init_ops = assemble(&[
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 0 },
        Op::PushByte { value: 7 },
        Op::CallProperty {
            index: idx(1), // helper
            num_args: 1,
        },
        Op::Pop,
        Op::GetLocal { index: 0 },
        Op::PushByte { value: 7 },
        Op::CallProperty {
            index: idx(2), // caller
            num_args: 1,
        },
        Op::ReturnValue,
    ]);
    // helper(n) = n + 100. The `getlocal0; pushscope` prologue makes `this`
    // (local 0) a used param, so helper's signature is [this, n] — the contiguous
    // shape the direct-call ABI requires.
    let helper_ops = assemble(&[
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 1 },
        Op::PushByte { value: 100 },
        Op::Add,
        Op::ReturnValue,
    ]);
    // caller(n) = this.helper(n).
    let caller_ops = assemble(&[
        Op::GetLocal { index: 0 },
        Op::PushScope,
        Op::GetLocal { index: 0 }, // receiver this
        Op::GetLocal { index: 1 }, // n
        Op::CallProperty {
            index: idx(1), // helper → CallMethod (sibling, co-compiled)
            num_args: 1,
        },
        Op::ReturnValue,
    ]);

    let int_param = || MethodParam {
        name: None,
        kind: idx(3), // multinames[3] = public::int
        default_value: None,
    };
    let one_int = || Method {
        name: idx(0),
        params: vec![int_param()],
        return_type: idx(0),
        flags: MethodFlags::empty(),
        body: None, // body index patched below
    };
    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![],
            uints: vec![],
            doubles: vec![],
            strings: vec![
                b"".to_vec(),
                b"helper".to_vec(),
                b"caller".to_vec(),
                b"int".to_vec(),
            ],
            namespaces: vec![Namespace::Package(idx(1))], // ns[1] = public ("")
            namespace_sets: vec![],
            multinames: vec![
                Multiname::QName {
                    namespace: idx(1),
                    name: idx(2),
                }, // [1] = public::helper
                Multiname::QName {
                    namespace: idx(1),
                    name: idx(3),
                }, // [2] = public::caller
                Multiname::QName {
                    namespace: idx(1),
                    name: idx(4),
                }, // [3] = public::int
            ],
        },
        methods: vec![
            Method {
                name: idx(0),
                params: vec![],
                return_type: idx(0),
                flags: MethodFlags::empty(),
                body: Some(idx(0)),
            },
            Method {
                body: Some(idx(1)),
                ..one_int()
            },
            Method {
                body: Some(idx(2)),
                ..one_int()
            },
        ],
        metadata: vec![],
        instances: vec![],
        classes: vec![],
        scripts: vec![Script {
            init_method: idx(0),
            traits: vec![
                Trait {
                    name: idx(1), // public::helper
                    kind: TraitKind::Method {
                        disp_id: 1,
                        method: idx(1),
                    },
                    metadata: vec![],
                    is_final: false,
                    is_override: false,
                },
                Trait {
                    name: idx(2), // public::caller
                    kind: TraitKind::Method {
                        disp_id: 2,
                        method: idx(2),
                    },
                    metadata: vec![],
                    is_final: false,
                    is_override: false,
                },
            ],
        }],
        method_bodies: vec![
            MethodBody {
                method: idx(0),
                max_stack: 2,
                num_locals: 1,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: init_ops,
                exceptions: vec![],
                traits: vec![],
            },
            MethodBody {
                method: idx(1),
                max_stack: 2,
                num_locals: 2,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: helper_ops,
                exceptions: vec![],
                traits: vec![],
            },
            MethodBody {
                method: idx(2),
                max_stack: 2,
                num_locals: 2,
                init_scope_depth: 0,
                max_scope_depth: 1,
                code: caller_ops,
                exceptions: vec![],
                traits: vec![],
            },
        ],
    };
    let mut abc_bytes = Vec::new();
    AbcWriter::new(&mut abc_bytes).write(abc).expect("write abc");

    let jit = run_and_get_jit(&build_swf(&abc_bytes));
    assert!(
        jit.direct_calls() >= 1,
        "the direct sibling-call path never fired (caller fell back to the trampoline)"
    );
}

/// Build an ABC whose script-init is `ops`, with one global slot trait `g`
/// (slot_id 1, default int 42) so typed property access resolves to slots.
fn build_slot_abc(ops: &[Op], max_stack: u32) -> Vec<u8> {
    let code = assemble(ops);
    let abc = AbcFile {
        major_version: 46,
        minor_version: 16,
        constant_pool: ConstantPool {
            ints: vec![42],                               // ints[1] = 42
            uints: vec![],
            doubles: vec![],
            strings: vec![b"".to_vec(), b"g".to_vec()],   // strings[1]="", [2]="g"
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
                name: idx(1), // public::g
                kind: TraitKind::Slot {
                    slot_id: 1,
                    type_name: idx(0), // "*" — accepts the int default
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

/// Load a hand-built SWF into a headless Player with a verifying JIT, run a frame,
/// and return the JIT so the caller can inspect its counters. Asserts the JIT
/// fired and agreed with the interpreter.
fn run_and_get_jit(swf_bytes: &[u8]) -> Rc<Jit2> {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let movie =
        SwfMovie::from_data(swf_bytes, "test.swf".to_string(), false, None).expect("valid swf");
    let jit = Rc::new(Jit2::new().with_verify(true));
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
    assert!(jit.hits() >= 1, "JIT never fired");
    assert_eq!(jit.mismatches(), 0, "JIT diverged from the interpreter");
    jit
}

/// Load a hand-built SWF into a headless Player with a verifying JIT and assert
/// the JIT fired and agreed with the interpreter.
fn run_and_check(swf_bytes: &[u8]) {
    run_and_get_jit(swf_bytes);
}

/// domainMemory (step 3): `si32(42 @0); li32(@0); returnvalue` → 42, run against
/// the default 1024-byte domain memory and matching the interpreter's direct
/// ByteArray access.
#[test]
fn domain_memory_store_load() {
    use Op::*;
    assert_jit_matches_interpreter(
        &[
            PushByte { value: 42 }, // value
            PushByte { value: 0 },  // address
            Si32,                   // dm[0..4] = 42
            PushByte { value: 0 },  // address
            Li32,                   // load dm[0..4]
            ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![],
    );
}

/// Float domainMemory round-trip (step 3): `sf64(3.5 @0); lf64(@0) + 0.0` → 3.5,
/// matching the interpreter (which uses `number_lossless`). The loaded raw Number
/// is laundered by the `+ 0.0` so it can be returned.
#[test]
fn float_domain_memory() {
    use Op::*;
    assert_jit_matches_interpreter(
        &[
            PushDouble { value: idx(1) }, // 3.5
            PushByte { value: 0 },
            Sf64, // dm[0..8] = 3.5
            PushByte { value: 0 },
            Lf64, // load it (raw)
            PushDouble { value: idx(2) }, // 0.0
            Add,
            ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![3.5, 0.0],
    );
}

/// `pushbyte 7; pushbyte 2; divide; returnvalue` → Number(3.5) (Divide is always
/// Number, even on int operands).
#[test]
fn divide_is_number() {
    assert_jit_matches_interpreter(
        &[
            Op::PushByte { value: 7 },
            Op::PushByte { value: 2 },
            Op::Divide,
            Op::ReturnValue,
        ],
        2,
        1,
        vec![],
        vec![],
    );
}
