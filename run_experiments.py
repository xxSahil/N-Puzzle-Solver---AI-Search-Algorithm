"""Experiment driver and visualization utilities for the n-puzzle project.

This module orchestrates experiments that run A* with multiple heuristics
across different puzzle sizes, collects results, computes summaries, and
produces CSVs and figures organized under a Results/PartX directory layout.

Author
------
Anton Nemchinski

API (at-a-glance)
------------------
- run_single_experiment(n, num_moves, seed, h_funcs) -> Dict[str, SearchResult]
    Run A* on a single randomized instance for all heuristics provided.
- run_multiple_experiments(n, num_moves, num_experiments, h_funcs)
    Produce results for many instances; prints progress and returns a dict
    mapping heuristic name -> list of SearchResult.
- calculate_statistics(results) -> pd.DataFrame
    Aggregate per-heuristic statistics (mean nodes, depth, runtime, b*, success_rate).
- create_and_save_comparison_table(results, n, base_path) -> pd.DataFrame
    Save a CSV comparing nodes expanded and branching factor per solution depth.
- save_image_table_nodes_and_bf(results, n, base_path) -> pd.DataFrame
    Produce the PNG table visual used in reports and a CSV summary.
- save_comparison_graphs(results, n, base_path)
    Save scatter and line plots (nodes vs depth, branching factor vs depth).
- ensure_figures_directory(n) -> str
    Create Results/PartX directories and return the path appropriate for n.
- run_part_1/2/3()
    Runners for 8-, 15-, and 24-puzzle experiments respectively. Each returns
    the raw results dict for that part.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Callable


# Project modules
from puzzle import Board
from search import astar, SearchResult
import heuristics
from metrics import branching_factor


def save_image_table_nodes_and_bf(results: Dict[str, List[SearchResult]], n: int, base_path: str):
    """Generate and save an image table like the textbook, for nodes and branching factor by depth and heuristic."""
    depth_groups = defaultdict(lambda: defaultdict(list))
    for h_name, h_results in results.items():
        for result in h_results:
            if result.solved:
                depth_groups[result.solution_depth][h_name].append(result)
    heuristics = list(results.keys())
    table_data = []
    for d in sorted(depth_groups.keys()):
        row = [d]
        # Nodes expanded for each heuristic
        for h_name in heuristics:
            vals = [r.nodes_expanded for r in depth_groups[d].get(h_name, [])]
            row.append(int(np.mean(vals)) if vals else '')
        # Branching factor for each heuristic
        for h_name in heuristics:
            vals = [branching_factor(r.nodes_expanded, r.solution_depth) for r in depth_groups[d].get(h_name, [])]
            row.append(round(np.mean(vals), 2) if vals else '')
        table_data.append(row)
    # Build column headers
    col_labels = ["d"] + [f"{h} Nodes" for h in heuristics] + [f"{h} BF" for h in heuristics]
    # Adjust figure size to fit table tightly
    fig_height = 0.6 * len(table_data) + 1.8
    fig_width = max(2 + 2 * len(heuristics), 8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    for col_idx in range(len(col_labels)):
        tbl.auto_set_column_width(col=col_idx)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.1, 1.1)
    # Style header row
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
        # Remove cell borders for a cleaner look
        cell.set_linewidth(0.5)
    # Center the table title and reduce whitespace
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.08)
    plt.title(f"A* Search Cost and Branching Factor ({n}x{n} Puzzle)", fontsize=14, pad=10)
    plt.savefig(f"{base_path}/table_nodes_bf_{n}x{n}.png", bbox_inches='tight', dpi=200)
    plt.close()
    # import matplotlib.ticker as ticker
    depth_groups = defaultdict(lambda: defaultdict(list))
    for h_name, h_results in results.items():
        for result in h_results:
            if result.solved:
                depth_groups[result.solution_depth][h_name].append(result.nodes_expanded)
    table_data = []
    heuristics = list(results.keys())
    for d in sorted(depth_groups.keys()):
        row = {'d': d}
        for h_name in heuristics:
            vals = depth_groups[d].get(h_name, [])
            row[h_name] = int(np.mean(vals)) if vals else ''
        table_data.append(row)
    df = pd.DataFrame(table_data)
    df.to_csv(f'{base_path}/summary_nodes_by_depth_{n}x{n}.csv', index=False)
    return df

def run_single_experiment(n: int, num_moves: int, seed: int, 
                         h_funcs: Dict[str, Callable]) -> Dict[str, SearchResult]:
    """Run experiment for a single puzzle instance with all heuristics."""
    board = Board.goal(n).randomize(num_moves, seed=seed)
    results = {}
    
    for h_name, h_func in h_funcs.items():
        result = astar(board, h_func)
        results[h_name] = result
        
    return results

def run_multiple_experiments(n: int, num_moves: int, num_experiments: int,
                           h_funcs: Dict[str, Callable]) -> Dict[str, List[SearchResult]]:
    """Run multiple experiments and collect results."""
    results = defaultdict(list)
    
    for seed in range(num_experiments):
        single_results = run_single_experiment(n, num_moves, seed, h_funcs)
        for h_name, result in single_results.items():
            results[h_name].append(result)
        if (seed + 1) % 10 == 0 or (seed + 1) == num_experiments:
            print(f"  Completed {seed + 1}/{num_experiments} experiments...")
    return dict(results)

def calculate_statistics(results: Dict[str, List[SearchResult]]) -> pd.DataFrame:
    """Calculate statistics from experimental results."""
    stats = {}
    
    for h_name, h_results in results.items():
        solved_results = [r for r in h_results if r.solved]
        if not solved_results:
            continue
            
        stats[h_name] = {
            'solution_depth_mean': int(np.mean([r.solution_depth for r in solved_results])),
            'nodes_expanded_mean': int(np.mean([r.nodes_expanded for r in solved_results])),
            'runtime_mean': np.mean([r.runtime for r in solved_results]).round(4),
            'branching_factor': np.mean([
                branching_factor(r.nodes_expanded, r.solution_depth) 
                for r in solved_results
            ]).round(3),
            'success_rate': round(len(solved_results) / len(h_results), 3)
        }

    return pd.DataFrame(stats).round(3)

def create_and_save_comparison_table(results: Dict[str, List[SearchResult]], n: int, base_path: str) -> pd.DataFrame:
    """Create and save a detailed comparison table."""
    depth_groups = defaultdict(lambda: defaultdict(list))
    
    for h_name, h_results in results.items():
        for result in h_results:
            if result.solved:
                depth_groups[result.solution_depth][h_name].append(result)
    
    table_data = []
    for d in sorted(depth_groups.keys()):
        row = {'Solution Depth': d}
        for h_name, results_at_depth in depth_groups[d].items():
            nodes = np.mean([r.nodes_expanded for r in results_at_depth])
            bf = np.mean([
                branching_factor(r.nodes_expanded, r.solution_depth)
                for r in results_at_depth
            ])
            row[f'{h_name} Nodes Expanded'] = int(nodes)
            row[f'{h_name} Branching Factor'] = round(bf, 2)
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    # Save as CSV only
    df.to_csv(f'{base_path}/comparison_table_{n}x{n}.csv', index=False)
    return df

def analyze_heuristic_domination(results: Dict[str, List[SearchResult]]) -> pd.DataFrame:
    """
    Analyze whether heuristics dominate each other.
    A heuristic h1 dominates h2 if:
    1. h1 expands fewer or equal nodes for all problems
    2. h1 expands strictly fewer nodes for at least one problem
    """
    heuristics = list(results.keys())
    domination_data = []
    
    for h1 in heuristics:
        for h2 in heuristics:
            if h1 == h2:
                continue
                
            # Match problems by solution depth for fair comparison
            h1_by_depth = defaultdict(list)
            h2_by_depth = defaultdict(list)
            
            for r in results[h1]:
                if r.solved:
                    h1_by_depth[r.solution_depth].append(r.nodes_expanded)
            for r in results[h2]:
                if r.solved:
                    h2_by_depth[r.solution_depth].append(r.nodes_expanded)
                    
            common_depths = set(h1_by_depth.keys()) & set(h2_by_depth.keys())
            if not common_depths:
                continue
                
            dominates = True
            strict_dominance = False
            
            for depth in common_depths:
                h1_nodes = np.mean(h1_by_depth[depth])
                h2_nodes = np.mean(h2_by_depth[depth])
                
                if h1_nodes > h2_nodes:
                    dominates = False
                    break
                if h1_nodes < h2_nodes:
                    strict_dominance = True
                    
            if dominates and strict_dominance:
                domination_data.append({
                    'dominating': h1,
                    'dominated': h2,
                    'common_problems': len(common_depths)
                })
    
    return pd.DataFrame(domination_data)

def save_comparison_graphs(results: Dict[str, List[SearchResult]], n: int, base_path: str):
    """Create and save comparison graphs for the results."""
    # Nodes vs Depth Plot
    plt.figure(figsize=(10, 6))
    for h_name, h_results in results.items():
        solved_results = [r for r in h_results if r.solved]
        if not solved_results:
            continue
            
        depths = [r.solution_depth for r in solved_results]
        nodes = [r.nodes_expanded for r in solved_results]
        plt.scatter(depths, nodes, label=h_name, alpha=0.5)
    
    plt.xlabel('Solution Depth')
    plt.ylabel('Nodes Expanded')
    plt.yscale('log')
    plt.legend()
    plt.title(f'{n}x{n} Puzzle: Solution Depth vs Nodes Expanded')
    plt.grid(True)
    plt.savefig(f'{base_path}/nodes_vs_depth_{n}x{n}.png')
    plt.close()
    
    # Branching Factor Plot
    plt.figure(figsize=(10, 6))
    depths = defaultdict(lambda: defaultdict(list))
    for h_name, h_results in results.items():
        for r in h_results:
            if r.solved:
                bf = branching_factor(r.nodes_expanded, r.solution_depth)
                depths[r.solution_depth][h_name].append(bf)
    
    for h_name in results.keys():
        x = sorted(depths.keys())
        y = [np.mean(depths[d][h_name]) for d in x if depths[d][h_name]]
        plt.plot(x, y, marker='o', label=h_name)
    
    plt.xlabel('Solution Depth')
    plt.ylabel('Effective Branching Factor')
    plt.title(f'{n}x{n} Puzzle: Effective Branching Factor vs Solution Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{base_path}/branching_factor_{n}x{n}.png')
    plt.close()

def ensure_figures_directory(n=None):
    """Create Part1, Part2, Part3 directories if they don't exist and return the correct one for n."""
    import os
    parent_dir = "Results"
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if n == 3:
        part_dir = os.path.join(parent_dir, "Part1")
    elif n == 4:
        part_dir = os.path.join(parent_dir, "Part2")
    elif n == 5:
        part_dir = os.path.join(parent_dir, "Part3")
    else:
        part_dir = os.path.join(parent_dir, "figures")
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)
    return part_dir

def run_part_1():
    # Run experiments for 8-puzzle
    """Run experiments for 8-puzzle."""
    """Run experiments for 8-puzzle."""
    print("\n=== Part 1: 8-puzzle Experiments ===")
    n = 3  # 8-puzzle is 3x3
    h_funcs = {
        'h1 (Misplaced)': heuristics.h1_misplaced,
        'h2 (Manhattan)': heuristics.h2_manhattan,
        'h3 (Linear Conflict)': heuristics.h3_linear_conflict
    }
    
    results = run_multiple_experiments(n, num_moves=20, num_experiments=100, h_funcs=h_funcs)
    figures_dir = ensure_figures_directory(n)
    
    print("\nStatistics:")
    stats = calculate_statistics(results)
    print(stats)
    stats.to_csv(f'{figures_dir}/statistics_8puzzle.csv')
    
    print("\nDetailed Performance Comparison by Solution Depth:")
    comparison = create_and_save_comparison_table(results, n, figures_dir)
    print(comparison)
    
    print("\nHeuristic Domination Analysis:")
    domination = analyze_heuristic_domination(results)
    print(domination)
    domination.to_csv(f'{figures_dir}/domination_8puzzle.csv')
    
    save_comparison_graphs(results, n, figures_dir)
    # Export all raw results for each heuristic
    for h_name, h_results in results.items():
        pd.DataFrame([
            {
                'solved': r.solved,
                'solution_depth': r.solution_depth,
                'nodes_expanded': r.nodes_expanded,
                'runtime': r.runtime
            } for r in h_results
        ]).to_csv(f'{figures_dir}/raw_results_{h_name.replace(" ", "_").replace("(", "").replace(")", "")}_8puzzle.csv', index=False)
    # Save PNG table at the end
    save_image_table_nodes_and_bf(results, n, figures_dir)
    return results

def run_part_2():
    # Run experiments for 15-puzzle
    """Run experiments for 15-puzzle."""
    """Run experiments for 15-puzzle."""
    print("\n=== Part 2: 15-puzzle Experiments ===")
    n = 4  # 15-puzzle is 4x4
    h_funcs = {
        'h1 (Misplaced)': heuristics.h1_misplaced,
        'h2 (Manhattan)': heuristics.h2_manhattan,
        'h3 (Linear Conflict)': heuristics.h3_linear_conflict
    }
    
    results = run_multiple_experiments(n, num_moves=20, num_experiments=100, h_funcs=h_funcs)
    figures_dir = ensure_figures_directory(n)
    
    print("\nStatistics:")
    stats = calculate_statistics(results)
    print(stats)
    stats.to_csv(f'{figures_dir}/statistics_15puzzle.csv')
    
    print("\nDetailed Performance Comparison by Solution Depth:")
    comparison = create_and_save_comparison_table(results, n, figures_dir)
    print(comparison)
    
    print("\nHeuristic Domination Analysis:")
    domination = analyze_heuristic_domination(results)
    print(domination)
    domination.to_csv(f'{figures_dir}/domination_15puzzle.csv')
    
    save_comparison_graphs(results, n, figures_dir)
    # Export all raw results for each heuristic
    for h_name, h_results in results.items():
        pd.DataFrame([
            {
                'solved': r.solved,
                'solution_depth': r.solution_depth,
                'nodes_expanded': r.nodes_expanded,
                'runtime': r.runtime
            } for r in h_results
        ]).to_csv(f'{figures_dir}/raw_results_{h_name.replace(" ", "_").replace("(", "").replace(")", "")}_15puzzle.csv', index=False)
    # Save PNG table at the end
    save_image_table_nodes_and_bf(results, n, figures_dir)
    return results

def run_part_3():
    # Run experiments for 24-puzzle
    """Run experiments for 24-puzzle."""
    """Run experiments for 24-puzzle."""
    print("\n=== Part 3: 24-puzzle Experiments ===")
    n = 5  # 24-puzzle is 5x5
    h_funcs = {
        'h1 (Misplaced)': heuristics.h1_misplaced,
        'h2 (Manhattan)': heuristics.h2_manhattan,
        'h3 (Linear Conflict)': heuristics.h3_linear_conflict
    }
    
    results = run_multiple_experiments(n, num_moves=20, num_experiments=100, h_funcs=h_funcs)
    figures_dir = ensure_figures_directory(n)
    
    print("\nStatistics:")
    stats = calculate_statistics(results)
    print(stats)
    stats.to_csv(f'{figures_dir}/statistics_24puzzle.csv')
    
    print("\nDetailed Performance Comparison by Solution Depth:")
    comparison = create_and_save_comparison_table(results, n, figures_dir)
    print(comparison)
    
    print("\nHeuristic Domination Analysis:")
    domination = analyze_heuristic_domination(results)
    print(domination)
    domination.to_csv(f'{figures_dir}/domination_24puzzle.csv')
    
    save_comparison_graphs(results, n, figures_dir)
    # Export all raw results for each heuristic
    for h_name, h_results in results.items():
        pd.DataFrame([
            {
                'solved': r.solved,
                'solution_depth': r.solution_depth,
                'nodes_expanded': r.nodes_expanded,
                'runtime': r.runtime
            } for r in h_results
        ]).to_csv(f'{figures_dir}/raw_results_{h_name.replace(" ", "_").replace("(", "").replace(")", "")}_24puzzle.csv', index=False)
    # Save PNG table at the end
    save_image_table_nodes_and_bf(results, n, figures_dir)
    return results

def compare_all_results(results_8: dict, results_15: dict, results_24: dict):
    """Create unified comparison of all puzzle sizes."""
    print("\n=== Unified Comparison Across Puzzle Sizes ===")
    
    sizes = {3: results_8, 4: results_15, 5: results_24}
    
    for n, results in sizes.items():
        print(f"\n{n}x{n} Puzzle Statistics:")
        print(calculate_statistics(results))
    
    # Create combined visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (n, results) in enumerate(sizes.items()):
        for h_name, h_results in results.items():
            solved_results = [r for r in h_results if r.solved]
            if not solved_results:
                continue
                
            depths = [r.solution_depth for r in solved_results]
            nodes = [r.nodes_expanded for r in solved_results]
            
            axes[idx].scatter(depths, nodes, label=h_name, alpha=0.5)
        
        axes[idx].set_xlabel('Solution Depth')
        axes[idx].set_ylabel('Nodes Expanded')
        axes[idx].set_yscale('log')
        axes[idx].legend()
        axes[idx].set_title(f'{n}x{n} Puzzle')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run all experiments
    results_8 = run_part_1()
    results_15 = run_part_2()
    results_24 = run_part_3()
    
    # Show unified comparison
    compare_all_results(results_8, results_15, results_24)
