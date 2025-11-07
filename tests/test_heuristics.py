# tests/test_heuristics.py
import pytest
from puzzle import Board
from heuristics import h1_misplaced, h2_manhattan, h3_linear_conflict

def test_heuristics_zero_on_goal():
    g3 = Board.goal(3)
    assert h1_misplaced(g3) == 0
    assert h2_manhattan(g3) == 0
    assert h3_linear_conflict(g3) == 0

def test_misplaced_and_manhattan_simple():
    b = Board.from_flat(3, [1,2,3,4,5,6,7,0,8])  # blank swapped with 8
    assert h1_misplaced(b) == 1
    assert h2_manhattan(b) == 1  # tile 8 is 1 move away

def test_linear_conflict_at_least_manhattan():
    # Classic linear conflict example (two tiles in same row reversed)
    # Goal row 0: 1 2 3
    # Put 2 and 1 swapped in row 0
    b = Board.from_rows([
        [2,1,3],
        [4,5,6],
        [7,8,0],
    ])
    man = h2_manhattan(b)
    lin = h3_linear_conflict(b)
    assert lin >= man
    # For this arrangement, 2 and 1 are in same goal row with reversed order → +2
    assert lin == man + 2
