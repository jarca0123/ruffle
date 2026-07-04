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
    AbcFile, ConstantPool, Index, Method, MethodBody, MethodFlags, Op, Script,
};
use swf::avm2::write::Writer as AbcWriter;
use swf::{DoAbc2, DoAbc2Flag, FileAttributes, Header, SwfStr, Tag};

fn idx<T>(i: u32) -> Index<T> {
    Index(i, PhantomData)
}

/// Assembles an ABC with a single script-init method:
/// `getlocal0; pushscope; pushbyte 3; pushbyte 4; addi; returnvalue` → 7.
fn build_abc() -> Vec<u8> {
    let ops = [
        Op::GetLocal { index: 0 }, // this
        Op::PushScope,
        Op::PushByte { value: 3 },
        Op::PushByte { value: 4 },
        Op::AddI,
        Op::ReturnValue,
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
            num_locals: 1,
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

#[test]
fn jit_matches_real_interpreter_on_loaded_method() {
    let swf_bytes = build_swf(&build_abc());
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

    assert!(
        jit.hits() >= 1,
        "JIT never fired on the loaded script-init method"
    );
    assert_eq!(
        jit.mismatches(),
        0,
        "JIT diverged from the real interpreter"
    );
}
