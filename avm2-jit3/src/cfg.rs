//! CFG analysis over the basic-block list `translate` produces.
//!
//! `emit_dispatch` currently lowers control flow as a state machine: `loop { block { br_table on
//! STATE } }`, so EVERY edge is one central indirect jump. Reconstructing structured control flow
//! (`block`/`loop`/`if`) lets edges become direct `br`s — or vanish, when the target can simply be
//! emitted inline. This module computes the dominator-tree facts that transform needs, and
//! [`analyze`] reports how much of the dispatch it could actually remove.

use crate::emit::{Block, Term};

/// The successors of a terminator, in branch order.
pub(crate) fn succs(term: &Term) -> Vec<usize> {
    match *term {
        Term::Return => vec![],
        Term::Jump(j) => vec![j],
        Term::Cond { on_true, on_false } => vec![on_true, on_false],
    }
}

/// Dominator-tree facts about a block list, all indexed by block.
pub(crate) struct Cfg {
    /// Reverse postorder position of each block; `usize::MAX` for unreachable blocks.
    pub rpo_num: Vec<usize>,
    /// Blocks in reverse postorder (reachable ones only) — the order a structured layout emits.
    pub rpo: Vec<usize>,
    /// Immediate dominator of each block (`idom[entry] == entry`); `usize::MAX` if unreachable.
    pub idom: Vec<usize>,
    /// Predecessor count, split by edge direction (forward = source precedes target in RPO).
    pub fwd_preds: Vec<usize>,
    /// Backedges `(source, target)`: a retreating edge whose target dominates its source.
    pub backedges: Vec<(usize, usize)>,
    /// Retreating edges that are NOT backedges — each one makes the CFG irreducible.
    pub irreducible: Vec<(usize, usize)>,
}

impl Cfg {
    /// A merge node has ≥2 forward predecessors, so it cannot be emitted inline at one of them —
    /// it needs a `block` scope the predecessors `br` forward to.
    pub(crate) fn is_merge(&self, b: usize) -> bool {
        self.fwd_preds[b] >= 2
    }

    /// A loop header is the target of a backedge, so it needs a `loop` scope.
    pub(crate) fn is_loop_header(&self, b: usize) -> bool {
        self.backedges.iter().any(|&(_, t)| t == b)
    }

    /// Whether `a` dominates `b` (walking `idom` up from `b`).
    pub(crate) fn dominates(&self, a: usize, b: usize) -> bool {
        let mut x = b;
        loop {
            if x == a {
                return true;
            }
            let up = self.idom[x];
            if up == x || up == usize::MAX {
                return false;
            }
            x = up;
        }
    }
}

/// Builds the dominator tree of `blocks` (entry = block 0) and classifies every edge.
pub(crate) fn build(blocks: &[Block]) -> Cfg {
    let n = blocks.len();
    // Reverse postorder via an iterative DFS (the block lists are deep — no recursion).
    let mut postorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack = vec![(0usize, 0usize)]; // (block, next successor to visit)
    visited[0] = true;
    while let Some(&mut (b, ref mut i)) = stack.last_mut() {
        let s = succs(&blocks[b].term);
        if *i < s.len() {
            let next = s[*i];
            *i += 1;
            if next < n && !visited[next] {
                visited[next] = true;
                stack.push((next, 0));
            }
        } else {
            postorder.push(b);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = postorder.iter().rev().copied().collect();
    let mut rpo_num = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b] = i;
    }

    // Predecessors (reachable sources only).
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        if !visited[b] {
            continue;
        }
        for s in succs(&blocks[b].term) {
            if s < n {
                preds[s].push(b);
            }
        }
    }

    // Iterative dominators (Cooper/Harvey/Kennedy) over RPO.
    let mut idom = vec![usize::MAX; n];
    idom[0] = 0;
    let intersect = |mut a: usize, mut b: usize, idom: &[usize], rpo_num: &[usize]| {
        while a != b {
            while rpo_num[a] > rpo_num[b] {
                a = idom[a];
            }
            while rpo_num[b] > rpo_num[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new = usize::MAX;
            for &p in &preds[b] {
                if idom[p] == usize::MAX {
                    continue; // not yet processed on this pass
                }
                new = if new == usize::MAX { p } else { intersect(new, p, &idom, &rpo_num) };
            }
            if new != usize::MAX && idom[b] != new {
                idom[b] = new;
                changed = true;
            }
        }
    }

    // Classify each edge: retreating (target does not follow source in RPO) vs forward.
    let mut fwd_preds = vec![0usize; n];
    let mut backedges = Vec::new();
    let mut irreducible = Vec::new();
    let cfg_dom = Cfg {
        rpo_num: rpo_num.clone(),
        rpo: rpo.clone(),
        idom: idom.clone(),
        fwd_preds: vec![0; n],
        backedges: Vec::new(),
        irreducible: Vec::new(),
    };
    for b in 0..n {
        if !visited[b] {
            continue;
        }
        for s in succs(&blocks[b].term) {
            if s >= n {
                continue;
            }
            if rpo_num[s] <= rpo_num[b] {
                // Retreating. A backedge iff the target dominates the source (a natural loop);
                // otherwise the loop has two entries and no `loop` scope can express it.
                if cfg_dom.dominates(s, b) {
                    backedges.push((b, s));
                } else {
                    irreducible.push((b, s));
                }
            } else {
                fwd_preds[s] += 1;
            }
        }
    }
    Cfg { rpo_num, rpo, idom, fwd_preds, backedges, irreducible }
}

/// A structured-control-flow tree — the relooper's output, which `emit` walks directly. Scopes
/// carry the block they stand for as a *label*; `emit` resolves each [`Node::Br`] to a branch
/// depth against the scope stack, so a mis-nesting declines instead of emitting a wrong jump.
pub(crate) enum Node {
    /// `block { … }`: a `Br(label)` inside exits it, landing on `label`'s code, which follows.
    Block { label: usize, body: Vec<Node> },
    /// `loop { … }`: a `Br(label)` inside re-enters it. `label` is the loop header.
    Loop { label: usize, body: Vec<Node> },
    /// Basic block `.0`: reload its `entry_depth` crossing values from the spill locals, then
    /// emit its straight-line ops.
    Code(usize),
    /// Spill the top `.0` operand-stack values into the spill locals, ahead of an edge. A wasm
    /// scope seals the operand stack, so values crossing an edge travel through locals — exactly
    /// as the `br_table` state machine does. `0` (the FlasCC/LLVM shape) emits nothing.
    Spill(usize),
    /// Branch to the enclosing scope labelled `.0`.
    Br(usize),
    /// A `Cond` terminator: pop the condition, spill `cross` crossing values (both successors
    /// share the same `entry_depth`), then run one arm. Both arms are terminal (every path
    /// bottoms out in a `Br` or a `Return`), so neither falls out of the `if`.
    Cond { cross: usize, then: Vec<Node>, els: Vec<Node> },
}

/// One entry of the wasm scope stack, innermost last. `If` carries no label — nothing branches
/// to it — but it still occupies a branch depth, so it must be on the stack.
pub(crate) enum Scope {
    Loop(usize),
    Block(usize),
    If,
}

/// The branch depth of the scope labelled `label` (0 = innermost). `None` if it is not in scope,
/// which would be a bug in [`structure`] — callers decline rather than emit a wrong branch.
pub(crate) fn depth_of(scopes: &[Scope], label: usize) -> Option<u32> {
    scopes
        .iter()
        .rposition(|s| matches!(*s, Scope::Loop(l) | Scope::Block(l) if l == label))
        .map(|i| (scopes.len() - 1 - i) as u32)
}

/// Rebuilds structured control flow from `blocks`, after Ramsey's *Beyond Relooper* (2022): walk
/// the dominator tree, opening a `loop` scope at each loop header and a `block` scope for each
/// dominated merge node, so every edge becomes a direct `br` — or vanishes, when the target has a
/// single forward predecessor and can simply be emitted inline.
///
/// Values crossing an edge on the operand stack travel through the spill locals ([`Node::Spill`]),
/// so a non-empty entry stack structures fine. That matters: FlasCC/LLVM output never crosses
/// (Lua: 0 of 168 methods), but the AS3 compiler does constantly — 165 of Starling's 715 methods,
/// carrying **73.6% of its edges**, so declining on it would leave the hot OO paths (`render`,
/// `hitTest`, `copyTo`) on the state machine.
///
/// `None` declines (caller falls back to the `br_table` state machine) only for shapes this
/// cannot express: **irreducible** control flow (a loop with two entries has no `loop` scope —
/// 2 edges in 14323 on Lua, 0 on Starling), a branch out of range, or a `Cond` whose successors
/// disagree on `entry_depth` (the shared-spill assumption the state machine also makes).
pub(crate) fn structure(blocks: &[Block]) -> Option<Vec<Node>> {
    let n = blocks.len();
    if n == 0 {
        return None;
    }
    if blocks.iter().flat_map(|b| succs(&b.term)).any(|s| s >= n) {
        return None; // a branch out of range — malformed, decline (as `emit_dispatch` does)
    }
    // One spill serves both arms of a `Cond`, so they must agree on how much crosses.
    if blocks.iter().any(|b| match b.term {
        Term::Cond { on_true, on_false } => blocks[on_true].entry_depth != blocks[on_false].entry_depth,
        _ => false,
    }) {
        return None;
    }
    let cfg = build(blocks);
    if !cfg.irreducible.is_empty() {
        return None;
    }
    // Dominator-tree children, each list in ascending RPO (the order `node_within` nests them).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &b in cfg.rpo.iter().skip(1) {
        children[cfg.idom[b]].push(b);
    }
    for c in &mut children {
        c.sort_by_key(|&x| cfg.rpo_num[x]);
    }
    let mut scopes = Vec::new();
    Some(do_tree(0, blocks, &cfg, &children, &mut scopes))
}

/// Emits `x` wrapped in a `loop` scope if it is a loop header (so its backedges can `br` to it).
fn do_tree(
    x: usize,
    blocks: &[Block],
    cfg: &Cfg,
    children: &[Vec<usize>],
    scopes: &mut Vec<Scope>,
) -> Vec<Node> {
    if cfg.is_loop_header(x) {
        scopes.push(Scope::Loop(x));
        let body = do_node(x, blocks, cfg, children, scopes);
        scopes.pop();
        vec![Node::Loop { label: x, body }]
    } else {
        do_node(x, blocks, cfg, children, scopes)
    }
}

/// Wraps `x`'s code in one `block` scope per dominated merge node, then emits `x` itself.
fn do_node(
    x: usize,
    blocks: &[Block],
    cfg: &Cfg,
    children: &[Vec<usize>],
    scopes: &mut Vec<Scope>,
) -> Vec<Node> {
    let merges: Vec<usize> = children[x].iter().copied().filter(|&y| cfg.is_merge(y)).collect();
    node_within(x, &merges, blocks, cfg, children, scopes)
}

/// Nests the merge nodes `ys` (ascending RPO) around `x`'s code, highest RPO OUTERMOST, each
/// followed by its own subtree — so a `br` from anywhere inside reaches the right one:
/// `block { block { <x> } <y0> } <y1>`.
fn node_within(
    x: usize,
    ys: &[usize],
    blocks: &[Block],
    cfg: &Cfg,
    children: &[Vec<usize>],
    scopes: &mut Vec<Scope>,
) -> Vec<Node> {
    let Some((&last, rest)) = ys.split_last() else {
        // Innermost: `x`'s own code, then its terminator.
        let mut out = vec![Node::Code(x)];
        match blocks[x].term {
            Term::Return => {} // the ops already emitted a `Return`
            Term::Jump(t) => {
                // Spill before the edge — an inlined target still reloads in its `Code`, so the
                // round-trip is uniform (and free when nothing crosses).
                out.push(Node::Spill(blocks[t].entry_depth));
                out.extend(do_branch(x, t, blocks, cfg, children, scopes));
            }
            Term::Cond { on_true, on_false } => {
                // Stack: [crossing…, cond]. The spill must precede the `if` — a wasm scope seals
                // everything below it, so the arms could not reach the crossing values.
                scopes.push(Scope::If);
                let then = do_branch(x, on_true, blocks, cfg, children, scopes);
                let els = do_branch(x, on_false, blocks, cfg, children, scopes);
                scopes.pop();
                out.push(Node::Cond { cross: blocks[on_true].entry_depth, then, els });
            }
        }
        return out;
    };
    scopes.push(Scope::Block(last));
    let body = node_within(x, rest, blocks, cfg, children, scopes);
    scopes.pop();
    let mut out = vec![Node::Block { label: last, body }];
    out.extend(do_tree(last, blocks, cfg, children, scopes));
    out
}

/// Lowers one edge. A backedge or a merge target needs a real `br`; anything else is the target's
/// sole forward predecessor, so the target is emitted right here and the edge costs nothing.
fn do_branch(
    source: usize,
    target: usize,
    blocks: &[Block],
    cfg: &Cfg,
    children: &[Vec<usize>],
    scopes: &mut Vec<Scope>,
) -> Vec<Node> {
    if cfg.rpo_num[target] <= cfg.rpo_num[source] || cfg.is_merge(target) {
        vec![Node::Br(target)]
    } else {
        do_tree(target, blocks, cfg, children, scopes)
    }
}

/// What a relooper would do to this method's edges. `inline` edges disappear entirely (the target
/// is emitted right there); `br` edges become one direct branch; `dispatch` edges are what would
/// still need the `br_table` state machine.
pub(crate) struct Stats {
    pub blocks: usize,
    pub edges: usize,
    pub inline: usize,
    pub br_fwd: usize,
    pub br_loop: usize,
    pub dispatch: usize,
    pub loops: usize,
    pub merges: usize,
    pub cross_blocks: usize,
}

/// Classifies every edge the way the structuring transform would lower it. See [`Stats`].
pub(crate) fn analyze(blocks: &[Block]) -> Stats {
    let cfg = build(blocks);
    let n = blocks.len();
    let (mut inline, mut br_fwd, mut br_loop, mut dispatch, mut edges) = (0, 0, 0, 0, 0);
    for b in 0..n {
        if cfg.rpo_num[b] == usize::MAX {
            continue;
        }
        for s in succs(&blocks[b].term) {
            if s >= n {
                continue;
            }
            edges += 1;
            if cfg.rpo_num[s] <= cfg.rpo_num[b] {
                if cfg.dominates(s, b) {
                    br_loop += 1; // `br` to the enclosing `loop`
                } else {
                    dispatch += 1; // irreducible — no structured form
                }
            } else if cfg.is_merge(s) {
                br_fwd += 1; // `br` out of the target's `block` scope
            } else {
                inline += 1; // sole forward predecessor → emit the target right here
            }
        }
    }
    Stats {
        blocks: n,
        edges,
        inline,
        br_fwd,
        br_loop,
        dispatch,
        loops: (0..n).filter(|&b| cfg.is_loop_header(b)).count(),
        merges: (0..n).filter(|&b| cfg.is_merge(b)).count(),
        cross_blocks: blocks.iter().filter(|b| b.entry_depth > 0).count(),
    }
}
