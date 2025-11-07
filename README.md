# Assignment 1 ProposedStructure

This project implements the A\* algorithm for the n-puzzle (8/15/24).  
Below is a brief overview of the main files and their roles.

## Files
- **Results/**
  - Contains raw results and figures for Part 1, 2, and 3 of the assignment

- **puzzle.py**

  - Defines the `Board` class (immutable puzzle state).
  - Methods: generate goal state, check if solved, generate neighbors, randomize state, visualize.

- **heuristics.py**

  - Contains heuristic functions (e.g., misplaced tiles, Manhattan distance, linear conflict).
  - All functions accept a `Board` and return an integer cost estimate.

- **search.py**

  - Implements the A\* search algorithm.
  - Returns solution path, nodes expanded, path length, and runtime statistics.

- **metrics.py**

  - Computes derived metrics like the effective branching factor (b\*).


- **run_experiments.py**

  - Command-line script to orchestrate experiments.
  - Generates scrambled puzzles, runs A\* with different heuristics, and logs results.

- **tests/**
  - Unit tests for components.


