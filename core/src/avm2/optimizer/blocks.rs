use crate::avm2::op::Op;

use std::cell::Cell;
use fnv::FnvHashSet;

// Represents non-overlapping slices of ops with
// a single entry point and many exit points.
// (as opposed to basic blocks, which have 1 entry and exit point)
#[derive(Debug)]
pub struct Block<'a, 'gc> {
    // The ops making up this block.
    pub ops: &'a [Cell<Op<'gc>>],

    // The index of the first op making up this Block.
    pub start_index: usize,
}

pub fn assemble_blocks<'a, 'gc>(
    code: &'a [Cell<Op<'gc>>],
    jump_targets: &FnvHashSet<usize>,
) -> (Vec<Block<'a, 'gc>>, Vec<u32>) {
    let mut block_list = Vec::with_capacity(2);
    let mut current_block_start = 0;

    for (i, op) in code.iter().enumerate() {
        let op = op.get();
        if matches!(
            op,
            Op::Jump { .. }
                | Op::ReturnVoid { .. }
                | Op::ReturnValue { .. }
                | Op::Throw
                | Op::LookupSwitch(_)
        ) || jump_targets.contains(&(i + 1))
        // The next op is a jump target
        {
            let block = Block {
                start_index: current_block_start,
                ops: &code[current_block_start..i + 1],
            };

            block_list.push(block);

            current_block_start = i + 1;
        }
    }

    // Table mapping each block's start op-index to its block index. Op indices are
    // dense (`0..code.len()`), so a `Vec` indexed by op-index is O(1) with **no
    // hashing** — vs a `HashMap<usize, usize>` on the default SipHash, whose per-jump
    // lookup dominated `process_jump`/verification at load. Jump targets are always
    // block leaders, so a lookup always lands on a set entry; `u32::MAX` marks a
    // non-leader op (never looked up).
    let mut op_index_to_block_index_table = vec![u32::MAX; code.len()];
    for (i, block) in block_list.iter().enumerate() {
        op_index_to_block_index_table[block.start_index] = i as u32;
    }

    (block_list, op_index_to_block_index_table)
}
