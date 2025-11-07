"""Small metrics utilities for analyzing search behavior.

Currently this module exposes a single utility used by the experiment
driver to compute the effective branching factor (b*), the branching
factor that would produce the observed number of expanded nodes at a
given solution depth in a uniform tree.

Author
------
Jordan Franschman

API (at-a-glance)
------------------
- branching_factor(nodes_expanded: int, solution_depth: int, tolerance: float = 0.01) -> float
  Numerically solve for b* in the equation: 1 + b* + b*^2 + ... + b*^d = nodes_expanded
  using binary search. Returns 0.0 for trivial cases.

"""

def branching_factor(nodes_expanded: int, solution_depth: int, tolerance: float = 0.01) -> float:

    if solution_depth == 0 or nodes_expanded <= 1:
        return 0.0
    
    low,high = 1.0, float(nodes_expanded)
    
    while high-low > tolerance:
        mid = (low + high) / 2.0
      
        if abs(mid - 1.0) < 1e-9:
            total = solution_depth + 1
        else:
            total = (mid**(solution_depth + 1) - 1) / (mid - 1)
        
        if total < nodes_expanded:
            low = mid
        else:
            high = mid
    
    return (low + high) / 2.0
