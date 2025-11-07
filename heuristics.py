"""Heuristic functions for the n-puzzle.

This module provides admissible heuristics used by the A* search driver:
- h1_misplaced: number of misplaced tiles (ignore blank)
- h2_manhattan: sum of Manhattan distances to goal positions
- h3_linear_conflict: Manhattan + 2 * (linear conflict pairs)

Author
------
Julian Rincon

API (at-a-glance)
------------------
- _goal_pos_map(n) -> Dict[int, (row, col)]
    Map tile value to its goal coordinates (excludes blank).
- h1_misplaced(board: Board) -> int
    Count of non-blank tiles not in goal position. O(n^2).
- h2_manhattan(board: Board) -> int
    Sum of Manhattan distances for all tiles. O(n^2).
- h3_linear_conflict(board: Board) -> int
    Manhattan plus linear conflict correction; admissible and
    typically stronger than Manhattan. O(n^2) with a small constant.

"""

from __future__ import annotations
from typing import Tuple, Dict
from puzzle import Board

def _goal_pos_map(n: int) -> Dict[int, Tuple[int, int]]:
    """
    Map each tile value -> (goal_row, goal_col).
    Tile 0 (blank) is excluded.
    Goal layout is (1..n*n-1, 0).
    """
    m: Dict[int, Tuple[int, int]] = {}
    for v in range(1, n * n):
        r, c = divmod(v - 1, n)
        m[v] = (r, c)
    return m

def h1_misplaced(board: Board) -> int:
    """Number of tiles not in their goal position (ignore blank)."""
    return sum(
        1
        for i, v in enumerate(board.tiles)
        if v != 0 and i != (v - 1)
    )

def h2_manhattan(board: Board) -> int:
    """Sum of Manhattan distances of each tile from its goal position."""
    n = board.n
    goal = _goal_pos_map(n)
    dist = 0
    for i, v in enumerate(board.tiles):
        if v == 0:
            continue
        r, c = divmod(i, n)
        gr, gc = goal[v]
        dist += abs(r - gr) + abs(c - gc)
    return dist
# Helper function for h3
def _linear_conflicts_in_line(line_vals, line_index, is_row, goal):
    """
    Count linear conflicts among tiles in one row/column.
    - Only tiles whose goal row (or column) is this line are considered.
    - A pair (a,b) is in conflict if both belong to this line in goal,
      but their order is reversed relative to goal columns (or rows).
    Returns the count of *conflicting pairs* (each adds +2 to heuristic).
    """
    conflicts = 0
    seq = []
    for idx_along, v in enumerate(line_vals):
        if v == 0:
            continue
        gr, gc = goal[v]
        if is_row:
            if gr == line_index:
                seq.append((idx_along, gc))
        else:
            if gc == line_index:
                seq.append((idx_along, gr))
    
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i][1] > seq[j][1]:
                conflicts += 1
    return conflicts

def h3_linear_conflict(board: Board) -> int:
    """
    Linear Conflict heuristic:
      h = Manhattan + 2 * (# of linear-conflict pairs in rows and columns)
    Admissible and consistent; stronger than pure Manhattan.
    """
    n = board.n
    goal = _goal_pos_map(n)
    man = h2_manhattan(board)

    # Row 
    row_conflicts = 0
    for r in range(n):
        start = r * n
        row_vals = board.tiles[start : start + n]
        row_conflicts += _linear_conflicts_in_line(row_vals, r, True, goal)

    # Column
    col_conflicts = 0
    for c in range(n):
        col_vals = board.tiles[c : n * n : n]
        col_conflicts += _linear_conflicts_in_line(col_vals, c, False, goal)

    return man + 2 * (row_conflicts + col_conflicts)
