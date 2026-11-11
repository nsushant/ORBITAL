"""Exact dominance index for large archives (private).

A forest of immutable bounding-volume trees, merged like a binary counter
(Bentley-Saxe).  Queries run in Numba and answer the two questions
:class:`mdls.core.Archive` needs:

    blocked(q)  -- does a stored point dominate q?
    victims(q)  -- which stored points does q dominate?

Every tree is static, so a build never has to rebalance and the insertion
order cannot degenerate the structure.  Results are identical to a linear
scan: node bounds are only used to discard subtrees that provably cannot
contain a match, and leaf candidates are filtered with the exact predicate.

This module is an implementation detail; it is not part of the public API.
"""

from __future__ import annotations

import numpy as np
from numba import njit

LEAF = 16      # points per leaf
_STACK = 256   # traversal stack; depth is at most ~2.5 * log2(n / LEAF)


@njit(cache=True)
def _build(pts, ids, leaf):
    """Build a static BVH; returns flat arrays (bounds, children, leaves, order).

    Nodes split at the midpoint of their widest dimension.  A split that would
    leave either side with less than a quarter of the points falls back to a
    count split, which keeps the tree balanced whatever the data looks like.
    """
    n = pts.shape[0]
    m = pts.shape[1]
    # Every leaf holds at least leaf // 4 points, so 2 * n // (leaf // 4) nodes
    # is a safe upper bound.
    cap = 2 * (n // max(leaf // 4, 1) + 2) + 8

    lo = np.empty((cap, m), np.float64)
    hi = np.empty((cap, m), np.float64)
    left = np.full(cap, -1, np.int64)
    right = np.full(cap, -1, np.int64)
    start = np.full(cap, -1, np.int64)
    end = np.full(cap, -1, np.int64)
    order = np.empty(n, np.int64)

    idx = np.arange(n)
    node_stack = np.empty(cap, np.int64)
    lo_stack = np.empty(cap, np.int64)
    hi_stack = np.empty(cap, np.int64)

    node_stack[0] = 0
    lo_stack[0] = 0
    hi_stack[0] = n
    top = 1
    n_nodes = 1
    pos = 0

    while top > 0:
        top -= 1
        node = node_stack[top]
        i0 = lo_stack[top]
        i1 = hi_stack[top]
        count = i1 - i0

        for j in range(m):
            vmin = pts[idx[i0], j]
            vmax = vmin
            for t in range(i0 + 1, i1):
                v = pts[idx[t], j]
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
            lo[node, j] = vmin
            hi[node, j] = vmax

        if count <= leaf:
            start[node] = pos
            for t in range(i0, i1):
                order[pos] = idx[t]
                pos += 1
            end[node] = pos
            continue

        dim = 0
        widest = hi[node, 0] - lo[node, 0]
        for j in range(1, m):
            width = hi[node, j] - lo[node, j]
            if width > widest:
                widest = width
                dim = j

        split = 0.5 * (lo[node, dim] + hi[node, dim])
        i, j = i0, i1 - 1
        while i <= j:
            while i <= j and pts[idx[i], dim] <= split:
                i += 1
            while i <= j and pts[idx[j], dim] > split:
                j -= 1
            if i < j:
                tmp = idx[i]
                idx[i] = idx[j]
                idx[j] = tmp
                i += 1
                j -= 1
        mid = i
        if mid - i0 < count // 4 or i1 - mid < count // 4:
            mid = i0 + count // 2

        left[node] = n_nodes
        right[node] = n_nodes + 1
        n_nodes += 2

        node_stack[top] = left[node]
        lo_stack[top] = i0
        hi_stack[top] = mid
        top += 1
        node_stack[top] = right[node]
        lo_stack[top] = mid
        hi_stack[top] = i1
        top += 1

    return (lo[:n_nodes].copy(), hi[:n_nodes].copy(), left[:n_nodes], right[:n_nodes],
            start[:n_nodes], end[:n_nodes], order)


@njit(cache=True)
def _exists_le(lo, hi, left, right, start, end, order, pts, q, eps_rel, stack):
    """True when a stored point is no worse than q in every objective."""
    m = q.shape[0]
    stack[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        pruned = False
        for j in range(m):
            scale = max(abs(lo[node, j]), abs(hi[node, j]))
            scale = max(scale, abs(q[j]))
            if lo[node, j] > q[j] + eps_rel * scale:
                pruned = True
                break
        if pruned:
            continue
        if left[node] < 0:
            for t in range(start[node], end[node]):
                p = order[t]
                no_worse = True
                for j in range(m):
                    tol = eps_rel * max(abs(pts[p, j]), abs(q[j]))
                    if pts[p, j] > q[j] + tol:
                        no_worse = False
                        break
                if no_worse:
                    return True
            continue
        stack[top] = left[node]
        top += 1
        stack[top] = right[node]
        top += 1
    return False


@njit(cache=True)
def _collect_ge(lo, hi, left, right, start, end, order, pts, ids, q, eps_rel, out, offset, stack):
    """Write the ids of stored points that q strictly dominates; return the count."""
    m = q.shape[0]
    stack[0] = 0
    top = 1
    k = offset
    while top > 0:
        top -= 1
        node = stack[top]
        pruned = False
        for j in range(m):
            scale = max(abs(lo[node, j]), abs(hi[node, j]))
            scale = max(scale, abs(q[j]))
            if hi[node, j] < q[j] - eps_rel * scale:
                pruned = True
                break
        if pruned:
            continue
        if left[node] < 0:
            for t in range(start[node], end[node]):
                p = order[t]
                no_worse = True
                better = False
                for j in range(m):
                    tol = eps_rel * max(abs(pts[p, j]), abs(q[j]))
                    if q[j] > pts[p, j] + tol:
                        no_worse = False
                        break
                    if q[j] < pts[p, j] - tol:
                        better = True
                if no_worse and better:
                    out[k] = ids[p]
                    k += 1
            continue
        stack[top] = left[node]
        top += 1
        stack[top] = right[node]
        top += 1
    return k - offset


class _Tree:
    """One immutable bounding-volume tree over a set of objective rows.

    The tree owns its points: a caller that keeps a live buffer must not hand
    it over without copying, which is what ``__init__`` does.
    """

    __slots__ = ("pts", "ids", "lo", "hi", "left", "right", "start", "end", "order")

    def __init__(self, pts, ids):
        self._setup(np.array(pts, dtype=np.float64, copy=True),
                    np.array(ids, dtype=np.int64, copy=True))

    @classmethod
    def _owning(cls, pts, ids):
        """Wrap arrays the caller will never mutate again, without copying."""
        tree = cls.__new__(cls)
        tree._setup(np.ascontiguousarray(pts, dtype=np.float64),
                    np.ascontiguousarray(ids, dtype=np.int64))
        return tree

    def _setup(self, pts, ids):
        self.pts, self.ids = pts, ids
        (self.lo, self.hi, self.left, self.right,
         self.start, self.end, self.order) = _build(pts, ids, LEAF)

    def __len__(self):
        return len(self.pts)


class _Forest:
    """Static trees of size 1, 2, 4, ...; insert merges like a binary counter."""

    __slots__ = ("m", "trees", "n", "_out", "_stack")

    def __init__(self, m, n=0):
        self.m = int(m)
        self.trees = []
        self.n = 0
        self._out = np.empty(max(int(n), 64), dtype=np.int64)
        self._stack = np.empty(_STACK, dtype=np.int64)

    @classmethod
    def build(cls, rows):
        """Build a forest over `rows`, one tree per set bit of its length."""
        rows = np.array(rows, dtype=np.float64, copy=True)
        n = len(rows)
        if n == 0:
            return cls(rows.shape[1] if rows.ndim == 2 else 1)
        forest = cls(rows.shape[1], n)
        level, start = 0, 0
        while start < n:
            if (n >> level) & 1:
                size = 1 << level
                forest.trees.append(
                    _Tree._owning(rows[start:start + size],
                                  np.arange(start, start + size, dtype=np.int64)))
                start += size
            else:
                forest.trees.append(None)
            level += 1
        forest.n = n
        return forest

    def __len__(self):
        return self.n

    def add(self, point, id):
        """Insert one point; amortized O(log n) rebuild work."""
        tree = _Tree(np.asarray(point, dtype=np.float64).reshape(1, -1),
                     np.array([id], dtype=np.int64))
        level = 0
        while level < len(self.trees) and self.trees[level] is not None:
            old = self.trees[level]
            tree = _Tree._owning(np.vstack((old.pts, tree.pts)),
                                 np.concatenate((old.ids, tree.ids)))
            self.trees[level] = None
            level += 1
        if level == len(self.trees):
            self.trees.append(tree)
        else:
            self.trees[level] = tree
        self.n += 1
        if self.n >= len(self._out):
            self._out = np.empty(max(2 * self.n, 64), dtype=np.int64)

    def blocked(self, q, eps_rel):
        """True when a stored point is no worse than q in every objective."""
        q = np.ascontiguousarray(q, dtype=np.float64)
        for tree in self.trees:
            if tree is None:
                continue
            if _exists_le(tree.lo, tree.hi, tree.left, tree.right,
                          tree.start, tree.end, tree.order, tree.pts,
                          q, eps_rel, self._stack):
                return True
        return False

    def victims(self, q, eps_rel):
        """Ids of stored points that q strictly dominates.

        The result is a view into a reusable buffer; consume it before the
        next query on this forest.
        """
        q = np.ascontiguousarray(q, dtype=np.float64)
        total = 0
        for tree in self.trees:
            if tree is None:
                continue
            total += _collect_ge(tree.lo, tree.hi, tree.left, tree.right,
                                 tree.start, tree.end, tree.order, tree.pts,
                                 tree.ids, q, eps_rel, self._out, total,
                                 self._stack)
        return self._out[:total]
