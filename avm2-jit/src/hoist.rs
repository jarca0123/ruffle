//! Loop-invariant `getslot` hoisting.
//!
//! Starling's hot vertex-transform loop re-reads six matrix fields
//! (`GetLocalValue(m); GetSlot(a/b/c/d/tx/ty)`) on EVERY iteration. Those reads
//! are invariant: the receiver local is never reassigned in the loop and no op
//! in the loop can write an object slot. This pass pulls each unique
//! `(receiver, slot)` read into a scratch local filled once in the loop's
//! preheader, and turns the in-loop pair into a plain local read.
//!
//! ## Safety
//! - A hoisted read runs SPECULATIVELY (even when the loop executes zero
//!   times), so it must not throw or run user code: only `GetSlot`s whose op
//!   index the verifier marked null-safe (receiver proven NOT NULL — see
//!   `VerifiedMethodInfo::null_safe_getslots`) are hoisted. A not-null slot
//!   read is a pure load.
//! - Invariance needs the loop body to be unable to change the slot: the body
//!   must contain no calls, property/slot writes, or ops that may run user
//!   code (`clobbers`), and the receiver local must not be written.
//! - Only the canonical ASC2 `while`/`for` shape is transformed:
//!   `E: Jump(C); B: body…; C: cond…; I: IfTrue/IfFalseBoxed(B)` with the
//!   entry jump IMMEDIATELY before the body. The preheader is inserted before
//!   `E`, so it dominates the loop and every branch target `>= E` shifts
//!   uniformly — no per-edge disambiguation.

use crate::lower::JitOp;

/// Max scratch locals a method may consume for hoisting (on top of
/// `num_locals` + the inline pass's callee locals; the web frame allows
/// `MAX_FRAME_SLOTS` total — the caller checks the final sum).
const MAX_HOISTS_PER_METHOD: u32 = 16;

/// Runs the pass over `ops` (must be free of switches/exceptions — the caller
/// gates). `null_safe` holds op indices (1:1 with the pre-pass `ops`) of
/// null-safe `GetSlot`s; `first_scratch` is the first free local index.
/// Returns how many scratch locals were consumed.
pub(crate) fn hoist_pass(
    ops: &mut Vec<JitOp>,
    null_safe: &[u32],
    first_scratch: u32,
    max_locals: u32,
) -> u32 {
    let mut null_safe: Vec<u32> = null_safe.to_vec();
    let mut used: u32 = 0;
    // Each successful hoist splices the vec; rescan from the loop's end.
    let mut scan_from = 0usize;
    while used < MAX_HOISTS_PER_METHOD {
        let Some(loop_shape) = find_canonical_loop(ops, scan_from) else {
            break;
        };
        let (entry, body, backedge) = loop_shape;
        scan_from = backedge + 1;
        if first_scratch + used >= max_locals {
            break;
        }
        let region = &ops[body..=backedge];
        if !region_is_invariant_safe(region) {
            continue;
        }
        let written = written_locals(region);
        // Unique invariant (receiver, slot) pairs whose GetSlot is null-safe.
        let mut pairs: Vec<(JitOp, u32)> = Vec::new();
        for (off, w) in region.windows(2).enumerate() {
            let (recv, slot) = match (w[0], w[1]) {
                (JitOp::GetLocalValue(r), JitOp::GetSlot(id))
                    if !written.contains(&r) =>
                {
                    (JitOp::GetLocalValue(r), id)
                }
                (JitOp::GetScriptGlobals(k), JitOp::GetSlot(id)) => {
                    (JitOp::GetScriptGlobals(k), id)
                }
                _ => continue,
            };
            let getslot_idx = (body + off + 1) as u32;
            if !null_safe.contains(&getslot_idx) {
                continue;
            }
            if !pairs.contains(&(recv, slot)) {
                pairs.push((recv, slot));
            }
        }
        let budget = (MAX_HOISTS_PER_METHOD - used)
            .min(max_locals - first_scratch - used) as usize;
        pairs.truncate(budget);
        if pairs.is_empty() {
            continue;
        }

        // Assign scratch locals and rewrite the in-loop pairs.
        let scratch_of = |i: usize| first_scratch + used + i as u32;
        for idx in body..backedge {
            let pair = match (ops[idx], ops[idx + 1]) {
                (r @ JitOp::GetLocalValue(_), JitOp::GetSlot(id))
                | (r @ JitOp::GetScriptGlobals(_), JitOp::GetSlot(id)) => (r, id),
                _ => continue,
            };
            if let Some(i) = pairs.iter().position(|p| *p == pair) {
                ops[idx] = JitOp::GetLocalValue(scratch_of(i));
                ops[idx + 1] = JitOp::Nop;
            }
        }

        // Preheader: `recv; GetSlot; SetLocalValue(scratch)` per pair, before E.
        let insert_len = pairs.len() * 3;
        let mut preheader = Vec::with_capacity(insert_len);
        for (i, (recv, slot)) in pairs.iter().enumerate() {
            preheader.push(*recv);
            preheader.push(JitOp::GetSlot(*slot));
            preheader.push(JitOp::SetLocalValue(scratch_of(i)));
        }
        ops.splice(entry..entry, preheader);

        // Every branch target at/after the insertion point shifts uniformly.
        for op in ops.iter_mut() {
            retarget(op, entry, insert_len);
        }
        for idx in null_safe.iter_mut() {
            if *idx as usize >= entry {
                *idx += insert_len as u32;
            }
        }
        used += pairs.len() as u32;
        scan_from += insert_len;
    }
    used
}

/// Finds the first canonical while-loop at/after `from`:
/// `E: Jump(C); B=E+1: …; I: IfTrueBoxed/IfFalseBoxed(B)` with `B <= C <= I`,
/// no other backedge inside `[B, I]`, and no branch from outside `[E, I]`
/// targeting the region's interior. Returns `(E, B, I)`.
fn find_canonical_loop(ops: &[JitOp], from: usize) -> Option<(usize, usize, usize)> {
    for i in from..ops.len() {
        let target = match ops[i] {
            JitOp::IfTrueBoxed(t) | JitOp::IfFalseBoxed(t) => t,
            _ => continue,
        };
        if target > i {
            continue; // forward branch, not a backedge
        }
        let body = target;
        if body == 0 {
            continue;
        }
        let entry = body - 1;
        let JitOp::Jump(cond) = ops[entry] else {
            continue;
        };
        if cond < body || cond > i {
            continue;
        }
        // No other backedge inside the region.
        let inner_backedge = (body..i).any(|j| {
            ops[j]
                .target()
                .is_some_and(|t| t <= j && t >= body)
        });
        if inner_backedge {
            continue;
        }
        // No branch from outside [entry, i] may target the region's interior
        // (the preheader must dominate the loop).
        let external_entry = ops.iter().enumerate().any(|(j, op)| {
            (j < entry || j > i)
                && op.target().is_some_and(|t| t >= body && t <= i)
        });
        if external_entry {
            continue;
        }
        return Some((entry, body, i));
    }
    None
}

/// Whether every op in the loop region is on the allow-list: unable to write
/// an object slot or run user code that could. Numeric inline ops are allowed
/// even though their non-numeric fallback helper could invoke `valueOf` — the
/// verifier types their operands numeric wherever it matters, and a `valueOf`
/// that mutates a hoisted slot mid-loop is accepted as out of contract.
fn region_is_invariant_safe(region: &[JitOp]) -> bool {
    region.iter().all(|op| {
        matches!(
            op,
            JitOp::GetLocalValue(_)
                | JitOp::SetLocalValue(_)
                | JitOp::StoreLocalValue(_)
                | JitOp::IncDecLocalIValue(_, _)
                | JitOp::PushIntValue(_)
                | JitOp::PushConst(_)
                | JitOp::PushString(_)
                | JitOp::GetScriptGlobals(_)
                | JitOp::GetSlot(_)
                | JitOp::DupValue
                | JitOp::SwapValue
                | JitOp::Pop
                | JitOp::Nop
                | JitOp::AddIBoxed
                | JitOp::SubtractIBoxed
                | JitOp::MultiplyIBoxed
                | JitOp::IncrementIBoxed
                | JitOp::DecrementIBoxed
                | JitOp::ArithInt(_)
                | JitOp::ArithNum(_)
                | JitOp::CmpNum(_)
                | JitOp::BitOpInt(_)
                | JitOp::CoerceInt(_)
                | JitOp::CoerceBool
                | JitOp::DmLoad(_)
                | JitOp::DmStore(_)
                | JitOp::DmLoadF(_)
                | JitOp::DmStoreF(_)
                | JitOp::Jump(_)
                | JitOp::IfTrueBoxed(_)
                | JitOp::IfFalseBoxed(_)
        )
    })
}

/// The set of locals the region writes.
fn written_locals(region: &[JitOp]) -> Vec<u32> {
    let mut out = Vec::new();
    for op in region {
        let w = match op {
            JitOp::SetLocalValue(i)
            | JitOp::StoreLocalValue(i)
            | JitOp::IncDecLocalIValue(i, _) => *i,
            _ => continue,
        };
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

/// Shifts `op`'s branch target by `by` when it lands at/after `at`.
fn retarget(op: &mut JitOp, at: usize, by: usize) {
    match op {
        JitOp::Jump(t)
        | JitOp::IfLt(t)
        | JitOp::IfGe(t)
        | JitOp::IfFalse(t)
        | JitOp::IfTrue(t)
        | JitOp::IfTrueBoxed(t)
        | JitOp::IfFalseBoxed(t) => {
            if *t >= at {
                *t += by;
            }
        }
        _ => {}
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn hoists_invariant_getslot_out_of_canonical_loop() {
        // E: Jump(C); B: local1.slot2 -> local4; i++; C: i < n?; IfTrue(B); ret
        let mut ops = vec![
            JitOp::Jump(6),                // 0: E -> C
            JitOp::GetLocalValue(1),       // 1: B (receiver)
            JitOp::GetSlot(2),             // 2:   null-safe
            JitOp::SetLocalValue(4),       // 3
            JitOp::IncDecLocalIValue(3, true), // 4: i++
            JitOp::Nop,                    // 5
            JitOp::GetLocalValue(3),       // 6: C
            JitOp::IfTrueBoxed(1),         // 7: backedge -> B
            JitOp::ReturnVoidBoxed(0),     // 8
        ];
        let used = hoist_pass(&mut ops, &[2], 10, 512);
        assert_eq!(used, 1);
        // Preheader before old E: recv; GetSlot; SetLocal(10).
        assert_eq!(
            &ops[0..3],
            &[JitOp::GetLocalValue(1), JitOp::GetSlot(2), JitOp::SetLocalValue(10)]
        );
        assert_eq!(ops[3], JitOp::Jump(9)); // C shifted by 3
        assert_eq!(ops[4], JitOp::GetLocalValue(10)); // in-loop pair rewritten
        assert_eq!(ops[5], JitOp::Nop);
        assert_eq!(ops[10], JitOp::IfTrueBoxed(4)); // backedge -> shifted B
    }

    #[test]
    fn refuses_written_receiver_and_calls() {
        // Receiver reassigned in the loop → no hoist.
        let mut ops = vec![
            JitOp::Jump(5),
            JitOp::GetLocalValue(1),
            JitOp::GetSlot(2),
            JitOp::SetLocalValue(1), // clobbers the receiver
            JitOp::Nop,
            JitOp::GetLocalValue(3),
            JitOp::IfTrueBoxed(1),
            JitOp::ReturnVoidBoxed(0),
        ];
        assert_eq!(hoist_pass(&mut ops.clone(), &[2], 10, 512), 0);
        // A call in the body → no hoist (could write any slot).
        ops[3] = JitOp::CallMethod(1, 0, false);
        assert_eq!(hoist_pass(&mut ops, &[2], 10, 512), 0);
    }

    #[test]
    fn refuses_non_null_safe_getslot() {
        let mut ops = vec![
            JitOp::Jump(5),
            JitOp::GetLocalValue(1),
            JitOp::GetSlot(2),
            JitOp::Pop,
            JitOp::Nop,
            JitOp::GetLocalValue(3),
            JitOp::IfTrueBoxed(1),
            JitOp::ReturnVoidBoxed(0),
        ];
        // Slot index NOT in the null-safe set → speculative read could throw.
        assert_eq!(hoist_pass(&mut ops, &[], 10, 512), 0);
    }
}
