import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure shinka is in python path
sys.path.append(os.getcwd())

try:
    from shinka.utils.load_df import load_programs_to_df
except ImportError:
    load_programs_to_df = None

RESULTS_DIR = "results"

def get_latest_run_dir(exp_dir):
    """Finds the latest run directory within an experiment directory, or returns exp_dir itself if it contains the DB."""
    # Check if DB exists directly in exp_dir (new behavior)
    if os.path.exists(os.path.join(exp_dir, "evolution_db.sqlite")):
        return exp_dir
    if os.path.exists(os.path.join(exp_dir, "evolution.db")):
        return exp_dir

    # Otherwise look for subdirectories (old behavior)
    run_dirs = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    
    # Sort by name (which acts as timestamp sort due to naming convention)
    # or modification time as fallback
    run_dirs.sort(key=lambda x: os.path.basename(x))
    return run_dirs[-1]

def summarize(exp_name=None):
    base_results_dir = RESULTS_DIR
    if exp_name:
        base_results_dir = os.path.join(RESULTS_DIR, exp_name)

    if not os.path.exists(base_results_dir):
        print(f"Results directory '{base_results_dir}' not found.")
        return

    # Check if we have tasks or variants directly
    experiments = [d for d in glob.glob(os.path.join(base_results_dir, "*")) if os.path.isdir(d)]
    experiments.sort()

    print(f"Summary for Experiment: {exp_name if exp_name else 'Root'}")
    print(f"{'Task Name':<40} | {'Run/Rnd':<20} | {'Gens':<5} | {'Best Score':<12} | {'Metric':<20} | {'Status'}")
    print("-" * 125)

    for exp_path in experiments:
        exp_basename = os.path.basename(exp_path)
        
        # Skip utility or cache dirs if any
        if exp_basename.startswith("__") or exp_basename.startswith("."):
            continue

        # Support nested structure: Task/Variant
        subdirs = [d for d in glob.glob(os.path.join(exp_path, "*")) if os.path.isdir(d)]
        has_named_variants = any(os.path.basename(d) in ["qwen", "gemini"] for d in subdirs)
        
        if has_named_variants:
            for var_path in subdirs:
                var_name = os.path.basename(var_path)
                if var_name in ["qwen", "gemini"]:
                    latest_run = get_latest_run_dir(var_path)
                    process_run(latest_run, f"{exp_basename} ({var_name})")
        else:
            latest_run = get_latest_run_dir(exp_path)
            process_run(latest_run, exp_basename)

def process_run(latest_run, display_name):
    if load_programs_to_df is None:
        print(f"{display_name:<40} | {'-':<20} | {'-':<5} | {'-':<12} | {'-':<20} | No load_df")
        return

    if not latest_run:
        print(f"{display_name:<40} | {'N/A':<20} | {'-':<5} | {'-':<12} | {'-':<20} | No runs found")
        return

    run_name = os.path.basename(latest_run)
    db_path = os.path.join(latest_run, "evolution_db.sqlite")
    if not os.path.exists(db_path):
        db_path = os.path.join(latest_run, "evolution.db")
    
    if not os.path.exists(db_path):
        print(f"{display_name:<40} | {run_name:<20} | {'-':<5} | {'-':<12} | {'-':<20} | No DB found")
        return

    try:
        df = load_programs_to_df(db_path)
        if df is None or df.empty:
            print(f"{display_name:<40} | {run_name:<20} | {'0':<5} | {'-':<12} | {'-':<20} | Empty DB")
            return
        
        # Filter for correct programs
        if 'correct' in df.columns:
            correct_df = df[df['correct'] == True]
        else:
            correct_df = df

        total_gens = df['generation'].max() if 'generation' in df.columns else 0
        
        if correct_df.empty:
            print(f"{display_name:<40} | {run_name:<20} | {total_gens:<5} | {'-':<12} | {'-':<20} | No valid solutions")
            return

        # Find best score
        best_idx = correct_df['combined_score'].idxmax()
        best_row = correct_df.loc[best_idx]
        best_score = best_row['combined_score']
        
        # Try to identify what the score represents
        metric_name = "combined_score"
        priority_metrics = ['benchmark_ratio', 'avg_benchmark_ratio', 'sum_radii', 'radii_sum', 'inv_c1', 'inv_outer_hex_side_length']
        for m in priority_metrics:
            if m in best_row and pd.notnull(best_row[m]):
                if abs(best_row[m] - best_score) < 1e-6:
                    metric_name = m
                    break
        
        print(f"{display_name:<40} | {run_name:<20} | {total_gens:<5} | {best_score:<12.6f} | {metric_name:<20} | OK")

    except Exception as e:
        print(f"{display_name:<40} | {run_name:<20} | {'-':<5} | {'-':<12} | {'-':<20} | Error: {str(e)[:20]}")

if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    summarize(exp)