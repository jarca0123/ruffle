//! Verified AVM2 `Op` stream → [`Block`] CFG translation.
//!
//! The stream is split into basic blocks (leaders = index 0, every branch target, and the
//! fallthrough after every branch), each block a straight-line body plus a [`Term`] telling
//! how it hands off control. Straight-line bodies cover: local moves, primitive/string
//! constants, `pop`/`dup`/`swap`/`pushscope`, null-safe `getslot`, `getproperty`,
//! `callproperty`/`callpropvoid`, the arithmetic/comparison/coerce ops (via the `binop`
//! /`unop` helpers), in-place `inc/declocal`, and `returnvalue`/`returnvoid`. Control flow
//! covers `jump`/`iftrue`/`iffalse`; `emit` renders the blocks as a `br_table` dispatch loop.
//!
//! REQUIREMENT: the operand stack is empty at every block boundary (a `jump`/fallthrough
//! leaves it empty; a conditional leaves exactly the branch condition, which the terminator
//! consumes). This holds for statement-level `if`/`while`/`for`; expression-level control
//! (ternary, `&&`/`||` mid-expression) leaves a value across the edge → declines. Anything
//! unhandled (a `LookupSwitch`, an unsupported op) also declines → the interpreter runs it.

use std::collections::BTreeSet;

use crate::emit::{Block, JitOp, Term};
use crate::helpers::{binop_code, mop_code, unop_code};
use crate::value::to_bits;
use ruffle_core::avm2::script::Script;
use ruffle_core::avm2::{Class, Op, Value};

/// Translates `ops` into basic blocks, or `None` to decline the whole method.
///
/// - `null_safe`: parsed-code op indices whose `getslot` the verifier proved non-null.
/// - `local_types`: each local slot's declared class (`this`/untyped = `None`); slot `1 + i`
///   is param `i`'s type. Seeds the operand-type tracker (single-block methods only).
pub fn translate<'gc>(
    ops: &[Op<'gc>],
    null_safe: &[u32],
    local_types: &[Option<Class<'gc>>],
) -> Option<Vec<Block>> {
    reset_decline_reason();
    if ops.is_empty() {
        return None;
    }
    let leaders = compute_leaders(ops)?;
    let num_blocks = leaders.len();
    let block_of = |target: usize| -> Option<usize> { leaders.binary_search(&target).ok() };

    // Only maintain the runtime local scope stack for methods that actually READ it
    // (`getscopeobject`). Everything else keeps the free `getlocal0; pushscope` prologue
    // (pushscope → a plain `pop`).
    let needs_scopes = ops.iter().any(|op| matches!(op, Op::GetScopeObject { .. }));

    // Operand-stack depth entering each block. Propagated forward from the entry (depth 0):
    // each terminator hands its crossing depth to its successors. Reducible AVM2 control flow
    // reaches every block via a lower-index predecessor first, so a single index-order pass
    // sets each `entry_depth[b]` before block `b` is generated (a back-edge to an already-set
    // header must agree — the verifier guarantees consistent stack heights).
    let mut entry_depth: Vec<Option<usize>> = vec![None; num_blocks];
    entry_depth[0] = Some(0);
    let mut blocks = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = leaders[b];
        let end = *leaders.get(b + 1).unwrap_or(&ops.len());
        let ed = entry_depth[b].unwrap_or(0);

        let mut out = Vec::new();
        // Seed the operand-type stack with the `ed` values that crossed from a predecessor
        // (types unknown → conservative). `emit` reloads them from the spill locals.
        let mut tys: Vec<Option<Class<'gc>>> = vec![None; ed];
        // Seed param types only for a single-block method (its entry types are exact). With
        // control flow a block may have multiple predecessors, so be conservative (all
        // unknown → typed returns always coerce, which is correct).
        let mut locals: Vec<Option<Class<'gc>>> = if num_blocks == 1 {
            local_types.to_vec()
        } else {
            vec![None; local_types.len()]
        };

        let mut term: Option<Term> = None;
        // Successor block(s) + the operand depth crossing to each (propagated after the block).
        let mut succ: Vec<(usize, usize)> = Vec::new();
        for i in start..end {
            match &ops[i] {
                Op::Jump { offset } => {
                    let t = block_of(*offset)?;
                    succ.push((t, tys.len())); // the whole operand stack crosses
                    term = Some(Term::Jump(t));
                    break;
                }
                Op::IfTrue { offset } => {
                    tys.pop()?; // the condition — consumed by `truthy`; the rest crosses
                    let d = tys.len();
                    let (ot, of) = (block_of(*offset)?, block_of(i + 1)?);
                    succ.push((ot, d));
                    succ.push((of, d));
                    term = Some(Term::Cond { on_true: ot, on_false: of });
                    break;
                }
                Op::IfFalse { offset } => {
                    tys.pop()?;
                    let d = tys.len();
                    let (ot, of) = (block_of(i + 1)?, block_of(*offset)?);
                    succ.push((ot, d));
                    succ.push((of, d));
                    term = Some(Term::Cond { on_true: ot, on_false: of });
                    break;
                }
                op @ (Op::ReturnValue { .. } | Op::ReturnVoid { .. }) => {
                    translate_op(op, i, null_safe, needs_scopes, &mut out, &mut tys, &mut locals)?;
                    term = Some(Term::Return);
                    break;
                }
                // `throw` diverges — a terminator like `returnvalue`. Pop the thrown value;
                // `JitOp::Throw` stashes the error + `Return`s (rest of the block is dead).
                Op::Throw => {
                    tys.pop()?; // the thrown value (consumed by `throw_value`)
                    out.push(JitOp::Throw);
                    term = Some(Term::Return);
                    break;
                }
                op => translate_op(op, i, null_safe, needs_scopes, &mut out, &mut tys, &mut locals)?,
            }
        }

        // No explicit terminator → implicit fallthrough to the next block; the whole stack
        // crosses the edge.
        let term = match term {
            Some(t) => t,
            None => {
                let t = block_of(end)?; // `end` is the next leader (else off-the-end → None)
                succ.push((t, tys.len()));
                Term::Jump(t)
            }
        };
        // Hand each successor its crossing depth; a disagreement (irreducible / verifier-
        // inconsistent) declines.
        for (s, d) in succ {
            match entry_depth[s] {
                None => entry_depth[s] = Some(d),
                Some(existing) if existing == d => {}
                Some(_) => {
                    set_decline("stack@edge_conflict");
                    return None;
                }
            }
        }
        blocks.push(Block { ops: out, term, entry_depth: ed });
    }
    Some(blocks)
}

/// Basic-block leaders: index 0, every branch target and post-branch index. `None` if the
/// method has control flow this translator can't structure (`LookupSwitch`).
fn compute_leaders(ops: &[Op]) -> Option<Vec<usize>> {
    let mut set = BTreeSet::new();
    set.insert(0);
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Jump { offset } | Op::IfTrue { offset } | Op::IfFalse { offset } => {
                if *offset > ops.len() {
                    return None;
                }
                set.insert(*offset);
                if i + 1 < ops.len() {
                    set.insert(i + 1);
                }
            }
            Op::LookupSwitch(_) => {
                record_decline(op); // multi-target — not structured here (logged for the report)
                return None;
            }
            _ => {}
        }
    }
    Some(set.into_iter().collect())
}

/// Translates one straight-line op into `out`, maintaining the operand-type stack `tys` and
/// the per-slot `locals`. `None` declines (and records the unsupported op for the log).
fn translate_op<'gc>(
    op: &Op<'gc>,
    i: usize,
    null_safe: &[u32],
    needs_scopes: bool,
    out: &mut Vec<JitOp>,
    tys: &mut Vec<Option<Class<'gc>>>,
    locals: &mut Vec<Option<Class<'gc>>>,
) -> Option<()> {
    match op {
        Op::GetLocal { index } => {
            out.push(JitOp::GetLocal(*index));
            tys.push(locals.get(*index as usize).copied().flatten());
        }
        Op::SetLocal { index } => {
            out.push(JitOp::SetLocal(*index));
            let t = tys.pop()?;
            if let Some(slot) = locals.get_mut(*index as usize) {
                *slot = t;
            }
        }
        // `storelocal` PEEKS (keeps the value on the stack) — `dup; setlocal`. Net stack
        // unchanged; the local takes the top's type.
        Op::StoreLocal { index } => {
            let t = *tys.last()?;
            out.push(JitOp::Dup);
            out.push(JitOp::SetLocal(*index));
            if let Some(slot) = locals.get_mut(*index as usize) {
                *slot = t;
            }
        }
        Op::GetSlot { index } => {
            // Null-safe (verifier-proven) → the pure no-throw helper; otherwise the
            // null-checking helper (throws #1009 on a null/primitive receiver) + BailIfError.
            if null_safe.contains(&(i as u32)) {
                out.push(JitOp::GetSlot(*index as u32));
            } else {
                out.push(JitOp::GetSlotChecked(*index as u32));
                out.push(JitOp::BailIfError);
            }
            tys.pop()?;
            tys.push(None);
        }
        // getproperty FAST (`obj[name]`, runtime name from the stack). The name is the single
        // runtime param; a runtime NAMESPACE (lazy ns) would need an extra stack slot → decline.
        Op::GetPropertyFast { multiname } => {
            if multiname.has_lazy_ns() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetPropertyFast(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // name
            tys.pop()?; // object
            tys.push(None);
        }
        Op::SetPropertyFast { multiname } => {
            if multiname.has_lazy_ns() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetPropertyFast(ptr));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop); // discard the (undefined) result
            tys.pop()?; // value
            tys.pop()?; // name
            tys.pop()?; // object
        }
        // getproperty (static multiname). NB: only `Static` — `GetPropertyFast` uses a LAZY
        // (runtime-name) multiname whose name comes from the stack, which `get_property`
        // can't resolve (it panics). The lazy guard below is belt-and-suspenders.
        Op::GetPropertyStatic { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetProperty(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?;
            tys.push(None);
        }
        Op::CallProperty { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallProperty(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?;
            tys.push(None);
        }
        Op::CallPropVoid { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallProperty(ptr, *num_args));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?;
        }
        // callnative: a baked native fn-pointer call; pops args + receiver, result if wanted.
        Op::CallNative { method, num_args, push_return_value } => {
            let ptr = *method as usize as u64;
            out.push(JitOp::CallNative(ptr, *num_args));
            out.push(JitOp::BailIfError);
            if !push_return_value {
                out.push(JitOp::Pop);
            }
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            if *push_return_value {
                tys.push(None);
            }
        }
        // callmethod (by disp-id): pops args + receiver; result pushed only if wanted.
        Op::CallMethod { index, num_args, push_return_value } => {
            out.push(JitOp::CallMethod(*index as u32, *num_args));
            out.push(JitOp::BailIfError);
            if !push_return_value {
                out.push(JitOp::Pop);
            }
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            if *push_return_value {
                tys.push(None);
            }
        }
        // newarray: pop `num_args` elements, push the Array (no throw).
        Op::NewArray { num_args } => {
            for _ in 0..*num_args {
                tys.pop()?;
            }
            out.push(JitOp::NewArray(*num_args));
            tys.push(None);
        }
        // istypelate / astypelate: pop `type`/`class` then `value`, push the result. Throwing.
        Op::IsTypeLate => {
            tys.pop()?; // type
            tys.pop()?; // value
            out.push(JitOp::IsTypeLate);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        Op::AsTypeLate => {
            tys.pop()?; // class
            tys.pop()?; // value
            out.push(JitOp::AsTypeLate);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        // construct: pop args + ctor, push `new ctor(args)` (may throw).
        Op::Construct { num_args } => {
            out.push(JitOp::Construct(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // ctor
            tys.push(None);
        }
        // constructprop: `new source.<mn>(args)`; pops args + source, pushes the object.
        Op::ConstructProp { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::ConstructProp(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // source
            tys.push(None);
        }
        // constructslot: `new (source.slot[index])(args)`; pops args + source, pushes the object.
        Op::ConstructSlot { index, num_args } => {
            out.push(JitOp::ConstructSlot(*index as u32, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // source
            tys.push(None);
        }
        // call: `function.call(receiver, args)`; pops args + receiver + function, pushes result.
        Op::Call { num_args } => {
            out.push(JitOp::Call(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            tys.pop()?; // function
            tys.push(None);
        }
        // applytype: `base.<T…>` (e.g. `Vector.<int>`); pops types + base, pushes the applied type.
        Op::ApplyType { num_types } => {
            out.push(JitOp::ApplyType(*num_types));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_types {
                tys.pop()?;
            }
            tys.pop()?; // base
            tys.push(None);
        }
        // newobject: build a dynamic object from `num_args` [name, value] pairs.
        Op::NewObject { num_args } => {
            out.push(JitOp::NewObject(*num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..(*num_args * 2) {
                tys.pop()?;
            }
            tys.push(None);
        }
        // newfunction: build a closure from the baked method + current scope (no stack input).
        Op::NewFunction { method } => {
            out.push(JitOp::NewFunction(method.as_ptr() as usize as u64));
            tys.push(None);
        }
        // in: `name in value`; pops value + name, pushes a Boolean.
        Op::In => {
            tys.pop()?; // value
            tys.pop()?; // name
            out.push(JitOp::In);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        // nextvalue/nextname/hasnext: the for..in enumerant ops; pop index + value, push one.
        Op::NextValue => {
            tys.pop()?; // index
            tys.pop()?; // value
            out.push(JitOp::NextValue);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        Op::NextName => {
            tys.pop()?;
            tys.pop()?;
            out.push(JitOp::NextName);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        Op::HasNext => {
            tys.pop()?;
            tys.pop()?;
            out.push(JitOp::HasNext);
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        // MOP (domain-memory) loads + sign-extends: pop 1, push 1.
        Op::Li8 => mop_load_op(out, tys, mop_code::LI8)?,
        Op::Li16 => mop_load_op(out, tys, mop_code::LI16)?,
        Op::Li32 => mop_load_op(out, tys, mop_code::LI32)?,
        Op::Lf32 => mop_load_op(out, tys, mop_code::LF32)?,
        Op::Lf64 => mop_load_op(out, tys, mop_code::LF64)?,
        Op::Sxi1 => mop_load_op(out, tys, mop_code::SXI1)?,
        Op::Sxi8 => mop_load_op(out, tys, mop_code::SXI8)?,
        Op::Sxi16 => mop_load_op(out, tys, mop_code::SXI16)?,
        // MOP stores: pop value + address, no result.
        Op::Si8 => mop_store_op(out, tys, mop_code::SI8)?,
        Op::Si16 => mop_store_op(out, tys, mop_code::SI16)?,
        Op::Si32 => mop_store_op(out, tys, mop_code::SI32)?,
        Op::Sf32 => mop_store_op(out, tys, mop_code::SF32)?,
        Op::Sf64 => mop_store_op(out, tys, mop_code::SF64)?,
        // kill: set local `index` to `undefined` (`op_kill`). Push undefined + store; net
        // stack unchanged.
        Op::Kill { index } => {
            push_const(out, tys, to_bits(Value::Undefined));
            out.push(JitOp::SetLocal(*index));
            tys.pop()?;
            if let Some(s) = locals.get_mut(*index as usize) {
                *s = None;
            }
        }
        // hasnext2: reads/writes locals `object_register`/`index_register` in the frame and
        // pushes a Boolean (no operand-stack input). Those two locals become int/object.
        Op::HasNext2 { object_register, index_register } => {
            out.push(JitOp::HasNext2(*object_register, *index_register));
            out.push(JitOp::BailIfError);
            if let Some(s) = locals.get_mut(*object_register as usize) {
                *s = None;
            }
            if let Some(s) = locals.get_mut(*index_register as usize) {
                *s = None;
            }
            tys.push(None);
        }
        // getouterscope: push the index-th captured scope's object (no stack input, no throw).
        Op::GetOuterScope { index } => {
            out.push(JitOp::OuterScope(*index as u32));
            tys.push(None);
        }
        // getscriptglobals: push a baked script's globals (may throw on lazy init).
        Op::GetScriptGlobals { script } => {
            // SAFETY: `Script` is a single-`Gc` newtype live for the method's run; erased for
            // baking, reversed by `helpers::script_globals`.
            let ptr = unsafe { core::mem::transmute::<Script<'gc>, *const ()>(*script) };
            out.push(JitOp::ScriptGlobals(ptr as usize as u64));
            out.push(JitOp::BailIfError);
            tys.push(None);
        }
        // newactivation: push a fresh activation object for the baked class (no throw).
        Op::NewActivation { activation_class } => {
            // SAFETY: as `Op::Coerce` — a live single-`Gc` `Class` handle.
            let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*activation_class) };
            out.push(JitOp::NewActivation(ptr as usize as u64));
            tys.push(None);
        }
        // deleteproperty (static multiname): pop receiver, push Boolean. Lazy names decline.
        Op::DeleteProperty { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::DeleteProperty(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // receiver
            tys.push(None); // Boolean result
        }
        // super ops (static multiname): resolve against the bound superclass. Lazy → decline.
        Op::GetSuper { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::GetSuper(ptr));
            out.push(JitOp::BailIfError);
            tys.pop()?; // receiver
            tys.push(None);
        }
        Op::SetSuper { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetSuper(ptr));
            out.push(JitOp::BailIfPerr);
            tys.pop()?; // value
            tys.pop()?; // receiver
        }
        Op::CallSuper { multiname, num_args } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::CallSuper(ptr, *num_args));
            out.push(JitOp::BailIfError);
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver
            tys.push(None);
        }
        Op::ConstructSuper { num_args } => {
            out.push(JitOp::ConstructSuper(*num_args));
            out.push(JitOp::BailIfError);
            out.push(JitOp::Pop); // discard the (undefined) result
            for _ in 0..*num_args {
                tys.pop()?;
            }
            tys.pop()?; // receiver (no result pushed)
        }
        // setproperty (static multiname): `object.set_property(mn, value)`. Pops value then
        // object; may throw (null receiver / setter) → BailIfError.
        Op::SetPropertyStatic { multiname } => {
            if multiname.has_lazy_component() {
                set_decline("lazy_multiname");
                return None;
            }
            let ptr = gc_arena::Gc::as_ptr(*multiname) as usize as u64;
            out.push(JitOp::SetProperty(ptr));
            out.push(JitOp::BailIfPerr);
            tys.pop()?; // value
            tys.pop()?; // object
        }
        // Slot writes (mode: 0 coerce / 1 coerce-int / 2 no-coerce).
        Op::SetSlot { index } => set_slot(out, tys, *index, 0)?,
        Op::SetSlotCoerceI { index } => set_slot(out, tys, *index, 1)?,
        Op::SetSlotNoCoerce { index } => set_slot(out, tys, *index, 2)?,
        Op::Add { .. } => bin(out, tys, binop_code::ADD)?,
        Op::Subtract { .. } => bin(out, tys, binop_code::SUBTRACT)?,
        Op::Multiply => bin(out, tys, binop_code::MULTIPLY)?,
        Op::Divide => bin(out, tys, binop_code::DIVIDE)?,
        Op::Modulo => bin(out, tys, binop_code::MODULO)?,
        Op::BitAnd => bin(out, tys, binop_code::BITAND)?,
        Op::BitOr => bin(out, tys, binop_code::BITOR)?,
        Op::BitXor => bin(out, tys, binop_code::BITXOR)?,
        Op::LShift => bin(out, tys, binop_code::LSHIFT)?,
        Op::RShift => bin(out, tys, binop_code::RSHIFT)?,
        Op::URShift => bin(out, tys, binop_code::URSHIFT)?,
        Op::Equals => bin(out, tys, binop_code::EQUALS)?,
        Op::StrictEquals => bin(out, tys, binop_code::STRICT_EQUALS)?,
        Op::LessThan => bin(out, tys, binop_code::LESS_THAN)?,
        Op::LessEquals => bin(out, tys, binop_code::LESS_EQUALS)?,
        Op::GreaterThan => bin(out, tys, binop_code::GREATER_THAN)?,
        Op::GreaterEquals => bin(out, tys, binop_code::GREATER_EQUALS)?,
        Op::AddI => bin(out, tys, binop_code::ADD_I)?,
        Op::SubtractI => bin(out, tys, binop_code::SUBTRACT_I)?,
        Op::MultiplyI => bin(out, tys, binop_code::MULTIPLY_I)?,
        Op::Negate => un(out, tys, unop_code::NEGATE)?,
        Op::Increment => un(out, tys, unop_code::INCREMENT)?,
        Op::Decrement => un(out, tys, unop_code::DECREMENT)?,
        Op::Not => un(out, tys, unop_code::NOT)?,
        Op::BitNot => un(out, tys, unop_code::BITNOT)?,
        Op::NegateI => un(out, tys, unop_code::NEGATE_I)?,
        Op::IncrementI => un(out, tys, unop_code::INCREMENT_I)?,
        Op::DecrementI => un(out, tys, unop_code::DECREMENT_I)?,
        Op::CoerceB => un(out, tys, unop_code::COERCE_B)?,
        Op::CoerceD => un(out, tys, unop_code::COERCE_D)?,
        Op::CoerceI => un(out, tys, unop_code::COERCE_I)?,
        Op::CoerceU => un(out, tys, unop_code::COERCE_U)?,
        Op::CoerceS => un(out, tys, unop_code::COERCE_S)?,
        Op::ConvertS => un(out, tys, unop_code::CONVERT_S)?,
        Op::CoerceO => un(out, tys, unop_code::COERCE_O)?,
        Op::CoerceA => {} // `coerce a` (to `*`) is the identity
        Op::Coerce { class } => {
            let top = tys.last_mut()?;
            *top = Some(*class);
            // SAFETY: `Class` is a live single-`Gc` handle for the method's run; erased for
            // baking, reversed by `helpers::coerce_return` (same as `CoerceReturn`).
            let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*class) };
            out.push(JitOp::Coerce(ptr as usize as u64));
            out.push(JitOp::BailIfError);
        }
        Op::Swap => {
            let b = tys.pop()?;
            let a = tys.pop()?;
            tys.push(b);
            tys.push(a);
            out.push(JitOp::Swap);
        }
        Op::IncLocal { index } => local_inplace(out, locals, *index, unop_code::INCREMENT)?,
        Op::DecLocal { index } => local_inplace(out, locals, *index, unop_code::DECREMENT)?,
        Op::IncLocalI { index } => local_inplace(out, locals, *index, unop_code::INCREMENT_I)?,
        Op::DecLocalI { index } => local_inplace(out, locals, *index, unop_code::DECREMENT_I)?,
        Op::PushInt { value } => push_const(out, tys, to_bits(Value::from(*value))),
        Op::PushUint { value } => push_const(out, tys, to_bits(Value::from(*value))),
        Op::PushDouble { value } => push_const(out, tys, to_bits(Value::from(*value))),
        Op::PushTrue => push_const(out, tys, to_bits(Value::from(true))),
        Op::PushFalse => push_const(out, tys, to_bits(Value::from(false))),
        Op::PushNull => push_const(out, tys, to_bits(Value::Null)),
        Op::PushUndefined => push_const(out, tys, to_bits(Value::Undefined)),
        Op::PushString { string } => push_const(out, tys, to_bits(Value::from(*string))),
        Op::Pop => {
            out.push(JitOp::Pop);
            tys.pop()?;
        }
        // pushscope: pops the object. If the method reads scopes (`getscopeobject`), push it
        // onto the local scope stack (null-check + may throw). Otherwise the scope-stack side
        // effect is unobserved → a plain `pop` (the common `getlocal0; pushscope` prologue).
        Op::PushScope => {
            if needs_scopes {
                out.push(JitOp::ScopePush);
                out.push(JitOp::BailIfPerr);
            } else {
                out.push(JitOp::Pop);
            }
            tys.pop()?;
        }
        // popscope: no operand-stack effect. Maintain the scope stack only when it matters.
        Op::PopScope => {
            if needs_scopes {
                out.push(JitOp::ScopePop);
            }
        }
        // getscopeobject: push the index-th local scope's object (no throw). Requires the
        // scope stack to be maintained (`needs_scopes`); decline otherwise.
        Op::GetScopeObject { index } => {
            if !needs_scopes {
                set_decline("no_scopes");
                return None;
            }
            out.push(JitOp::GetScope(*index as u32));
            tys.push(None);
        }
        Op::Dup => {
            out.push(JitOp::Dup);
            let t = *tys.last()?;
            tys.push(t);
        }
        Op::ReturnValue { return_type } => {
            let top = tys.pop()?;
            match return_type {
                None => out.push(JitOp::ReturnValue),
                Some(rt) if top == Some(*rt) => out.push(JitOp::ReturnValue),
                Some(rt) => {
                    // SAFETY: same as `Op::Coerce` — a live single-`Gc` `Class` handle.
                    let ptr = unsafe { core::mem::transmute::<Class<'gc>, *const ()>(*rt) };
                    out.push(JitOp::CoerceReturn(ptr as usize as u64));
                }
            }
        }
        Op::ReturnVoid { return_type } => {
            // A typed `returnvoid` yields the type's DEFAULT value (mirrors `return_void`):
            // numeric → int 0, boolean → false, void → undefined, else → null. Compute it at
            // compile time and return that constant; untyped → plain `returnvoid` (undefined).
            match return_type {
                None => out.push(JitOp::ReturnVoid),
                Some(rt) => {
                    let default = if rt.is_builtin_void() {
                        to_bits(Value::Undefined)
                    } else if rt.is_builtin_numeric() {
                        to_bits(Value::from(0))
                    } else if rt.is_builtin_boolean() {
                        to_bits(Value::from(false))
                    } else {
                        to_bits(Value::Null)
                    };
                    out.push(JitOp::PushBits(default));
                    out.push(JitOp::ReturnValue);
                }
            }
        }
        other => {
            record_decline(other);
            return None;
        }
    }
    Some(())
}

fn push_const<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Option<Class<'gc>>>, bits: u64) {
    out.push(JitOp::PushBits(bits));
    tys.push(None);
}

/// A dynamic binary op: pops two operands, pushes one result of unknown class. The `binop`
/// helper can coerce (`valueOf`) → throw, so a `BailIfError` follows.
fn bin<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Option<Class<'gc>>>, code: i32) -> Option<()> {
    tys.pop()?;
    tys.pop()?;
    tys.push(None);
    out.push(JitOp::BinOp(code));
    out.push(JitOp::BailIfError);
    Some(())
}

/// A dynamic unary op: pops one operand, pushes one result of unknown class.
fn un<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Option<Class<'gc>>>, code: i32) -> Option<()> {
    tys.pop()?;
    tys.push(None);
    out.push(JitOp::UnOp(code));
    out.push(JitOp::BailIfError);
    Some(())
}

/// A MOP load / sign-extend: pops the address (or value), pushes one result. May throw.
fn mop_load_op<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Option<Class<'gc>>>, code: i32) -> Option<()> {
    tys.pop()?;
    tys.push(None);
    out.push(JitOp::MopLoad(code));
    out.push(JitOp::BailIfError);
    Some(())
}

/// A MOP store: pops the address then the value, no result (discards the undefined). May throw.
fn mop_store_op<'gc>(out: &mut Vec<JitOp>, tys: &mut Vec<Option<Class<'gc>>>, code: i32) -> Option<()> {
    tys.pop()?; // address
    tys.pop()?; // value
    out.push(JitOp::MopStore(code));
    out.push(JitOp::BailIfError);
    out.push(JitOp::Pop); // discard the (undefined) result
    Some(())
}

/// A slot write: pops `value` then `object`, no result. Guarded by a `BailIfError`.
fn set_slot<'gc>(
    out: &mut Vec<JitOp>,
    tys: &mut Vec<Option<Class<'gc>>>,
    index: usize,
    mode: i32,
) -> Option<()> {
    tys.pop()?; // value
    tys.pop()?; // object
    out.push(JitOp::SetSlot(index as u32, mode));
    out.push(JitOp::BailIfPerr);
    Some(())
}

/// In-place `inclocal`/`declocal`: `local[i] = unop(local[i], code)`, no net stack effect.
fn local_inplace<'gc>(
    out: &mut Vec<JitOp>,
    locals: &mut [Option<Class<'gc>>],
    index: u32,
    code: i32,
) -> Option<()> {
    out.push(JitOp::GetLocal(index));
    out.push(JitOp::UnOp(code));
    out.push(JitOp::BailIfError);
    out.push(JitOp::SetLocal(index));
    if let Some(slot) = locals.get_mut(index as usize) {
        *slot = None;
    }
    Some(())
}

/// Records an unsupported op for the decline log — the first sighting of each op variant is
/// logged (to stderr natively / the browser console on web) so we can see, from real
/// content, which ops to implement next. Subsequent sightings just bump a counter.
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};

thread_local! {
    /// Interns op-name strings to `&'static str` (bounded: ~one leak per distinct op
    /// variant) so the compile-decline reason can flow into the cache + profiler.
    static INTERNED: RefCell<HashMap<String, &'static str>> = RefCell::new(HashMap::new());
    /// The reason `translate` most recently declined — an `op:<Variant>` for a genuinely
    /// unsupported op, or the default `shape/guard` for a structural decline (non-empty
    /// operand stack across an edge, a guarded op like non-null-safe `getslot`, a lazy
    /// multiname, …). Read by `try_compile` after a `None` return.
    static LAST_DECLINE: Cell<&'static str> = const { Cell::new("shape/guard") };
}

/// Interns `name` into a process-lifetime `&'static str` (leaks once per distinct string).
fn intern(name: &str) -> &'static str {
    INTERNED.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(s) = m.get(name) {
            return *s;
        }
        let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
        m.insert(name.to_string(), leaked);
        leaked
    })
}

/// Resets the decline reason to the structural default; called at the top of `translate`.
pub(crate) fn reset_decline_reason() {
    LAST_DECLINE.with(|c| c.set("shape/guard"));
}

/// Labels the current structural decline (sub-buckets `shape/guard` for the profiler).
fn set_decline(reason: &'static str) {
    LAST_DECLINE.with(|c| c.set(reason));
}

/// The reason `translate` last declined (see [`LAST_DECLINE`]). Read by `try_compile`.
pub(crate) fn last_decline_reason() -> &'static str {
    LAST_DECLINE.with(|c| c.get())
}

fn record_decline(op: &Op) {
    thread_local! {
        static SEEN: RefCell<BTreeMap<String, u64>> = const { RefCell::new(BTreeMap::new()) };
    }
    // Variant name = the leading identifier of the `Debug` form (`Foo { .. }` → `Foo`).
    let dbg = format!("{op:?}");
    let name = dbg
        .split(|c: char| c == ' ' || c == '{' || c == '(')
        .next()
        .unwrap_or("?")
        .to_string();
    // Record the specific op as the current decline reason (frequency-weighted profiler).
    LAST_DECLINE.with(|c| c.set(intern(&format!("op:{name}"))));
    SEEN.with(|m| {
        let mut m = m.borrow_mut();
        let count = m.entry(name.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            crate::runner::log_decline(&name);
        }
    });
}
