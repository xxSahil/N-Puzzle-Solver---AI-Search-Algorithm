# tests/test_puzzle.py
import pytest
from puzzle import Board, manual_play


# ---------- Construction ----------

def test_goal_construction_and_identity():
    g3 = Board.goal(3)
    assert g3.n == 3
    assert g3.tiles == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert g3.zero == 8
    assert g3.is_goal() is True

def test_from_flat_validates_length_and_contents():
    with pytest.raises(ValueError):
        Board.from_flat(3, [1,2,3,4,5,6,7,8])  # too short
    with pytest.raises(ValueError):
        Board.from_flat(3, [0,0,1,2,3,4,5,6,7])  # duplicate 0
    ok = Board.from_flat(3, [1,2,3,4,5,6,7,8,0])
    assert ok.is_goal()

def test_from_rows_calls_validation_and_builds_board():
    b = Board.from_rows([[1,2,3],[4,5,6],[7,8,0]])
    assert b.n == 3 and b.zero == 8 and b.is_goal()
    # ragged rows -> n=3 but len(flat)=8 -> triggers from_flat length check
    with pytest.raises(ValueError):
        Board.from_rows([[1,2,3],[4,5,6],[7,8]])


# ---------- Neighbors ----------

def test_neighbors_corner_has_two():
    g = Board.goal(3)  # blank bottom-right corner
    nbrs = list(g.neighbors())
    assert len(nbrs) == 2
    # original remains unchanged (immutability check)
    assert g.tiles == (1,2,3,4,5,6,7,8,0)

def test_neighbors_next_state_degree_is_three_or_four():
    g = Board.goal(3)
    first = next(g.neighbors())         # Up from (2,2) -> blank to index 5
    deg = len(list(first.neighbors()))
    assert deg in (3, 4)  # edge → 3, center → 4 (depends on first move order)

def test_neighbors_make_valid_single_swaps_only():
    g = Board.goal(3)
    for nb in g.neighbors():
        # tiles differ by exactly one swap with the blank
        diffs = [i for i,(a,b) in enumerate(zip(g.tiles, nb.tiles)) if a != b]
        assert len(diffs) == 2
        assert g.tiles[diffs[0]] == nb.tiles[diffs[1]]
        assert g.tiles[diffs[1]] == nb.tiles[diffs[0]]
        # blank moved to a Manhattan-adjacent index
        z0, z1 = g.zero, nb.zero
        r0,c0 = divmod(z0, g.n)
        r1,c1 = divmod(z1, g.n)
        assert abs(r0-r1) + abs(c0-c1) == 1


# ---------- Solvability ----------

def test_solvability_rules_3x3():
    solvable = Board.from_rows([[1,2,3],[4,5,6],[7,8,0]])
    unsolvable = Board.from_rows([[1,2,3],[4,5,6],[8,7,0]])  # odd n; swap -> odd inversions
    assert solvable.is_solvable() is True
    assert unsolvable.is_solvable() is False

def test_solvability_rules_4x4():
    g = Board.goal(4)
    assert g.is_solvable() is True
    # swap two numbered tiles in goal (keep blank at end) -> should become unsolvable
    bad = Board.from_flat(4, [2,1,3,4,5,6,7,8,9,10,11,12,13,14,15,0])
    assert bad.is_solvable() is False


# ---------- Utilities ----------

def test_visualize_line_count_and_spacing():
    g = Board.goal(3)
    s = g.visualize()
    assert s.count("\n") == 2
    assert "  " in s.splitlines()[-1]  # last row shows blank as spaces

def test_randomize_is_deterministic_with_seed_and_solvable():
    g = Board.goal(3)
    a = g.randomize(25, seed=123)
    b = g.randomize(25, seed=123)
    c = g.randomize(25, seed=124)
    assert a.tiles == b.tiles
    assert a.tiles != c.tiles
    assert a.is_solvable() and b.is_solvable() and c.is_solvable()


# ---------- Hashability / Equality ----------

def test_hash_and_equality_work_in_sets_and_dicts():
    a = Board.goal(3)
    b = Board.from_flat(3, list(a.tiles))
    assert a == b
    s = {a}
    assert b in s
    d = {a: "ok"}
    assert d[b] == "ok"


if __name__ == "__main__":
    pytest.main()# puzzle.py

    ### Manual Testing Code ###
    solved_board = Board.goal(3)
    print("Solved 3x3 Board:")
    print(solved_board.visualize())
    print("Board is goal?", solved_board.is_goal())
    for nbr in solved_board.neighbors():
        print("\nNeighbor Board:")
        print(nbr.visualize())
        print("Board is goal?", nbr.is_goal())
        print("Board is solvable?", nbr.is_solvable())
        print("Board hash:", hash(nbr))
    random_board = solved_board.randomize(10, seed=42)
    print("\nRandomized Board after 10 moves:")

    manual_board = Board.from_flat(4, [1,2,3,4,5,6,0,8,9,10,7,11,13,14,15,12])
    print("\nManual 4x4 Board:")
    print(manual_board.visualize())


    manual_rows_board_5 = Board.from_rows([[1,2,3,4,5],[6,7,8,9,10],[11,12,0,13,14],[16,17,18,19,15],[21,22,23,24,20]])
    print("\nManual 5x5 Board from Rows:")
    print(manual_rows_board_5.visualize())

    n = int(input("Enter board size n for an n x n puzzle (e.g., 3 for 8-puzzle): "))
    r = int(input("Enter number of random moves to shuffle the board: "))
    manual_play(n,r)