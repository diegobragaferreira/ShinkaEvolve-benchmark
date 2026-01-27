import os
import sys
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

# Ensure shinka is in python path
sys.path.append(os.getcwd())

from shinka.utils.load_df import load_programs_to_df

RESULTS_DIR = "results"

def get_latest_run_dir(exp_dir):
    """Finds the latest run directory within an experiment directory, or returns exp_dir itself if it contains the DB."""
    # Check if DB exists directly in exp_dir (new behavior)
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

def summarize():
    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return

    experiments = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    experiments.sort()

    summary_data = []

    print(f"{'Task Name':<40} | {'Run Date':<20} | {'Gens':<5} | {'Best Score':<12} | {'Metric':<20} | {'Status'}")
    print("-" * 120)

    for exp_path in experiments:
        exp_name = os.path.basename(exp_path)
        
        # Skip utility or cache dirs if any
        if exp_name.startswith("__") or exp_name.startswith("."):
            continue

        latest_run = get_latest_run_dir(exp_path)
        if not latest_run:
            print(f"{exp_name:<40} | {'N/A':<20} | {'-':<5} | {'-':<12} | {'-':<20} | No runs found")
            continue

        run_name = os.path.basename(latest_run)
        db_path = os.path.join(latest_run, "evolution.db")
        
        if not os.path.exists(db_path):
            print(f"{exp_name:<40} | {run_name:<20} | {'-':<5} | {'-':<12} | {'-':<20} | No DB found")
            continue

        try:
            df = load_programs_to_df(db_path)
            if df is None or df.empty:
                print(f"{exp_name:<40} | {run_name:<20} | {'0':<5} | {'-':<12} | {'-':<20} | Empty DB")
                continue
            
            # Filter for correct programs
            if 'correct' in df.columns:
                correct_df = df[df['correct'] == True]
            else:
                correct_df = df # Assume all are correct if column missing (unlikely)

            total_gens = df['generation'].max()
            
            if correct_df.empty:
                print(f"{exp_name:<40} | {run_name:<20} | {total_gens:<5} | {'-':<12} | {'-':<20} | No valid solutions")
                continue

            # Find best score
            # combined_score is the fitness.
            best_idx = correct_df['combined_score'].idxmax()
            best_row = correct_df.loc[best_idx]
            best_score = best_row['combined_score']
            
            # Try to identify what the score represents from metadata or metrics
            metric_name = "combined_score"
            # Common metrics in our configs: benchmark_ratio, inv_c1, sum_radii, etc.
            # We can check columns.
            priority_metrics = ['benchmark_ratio', 'avg_benchmark_ratio', 'sum_radii', 'radii_sum', 'inv_c1', 'inv_outer_hex_side_length']
            for m in priority_metrics:
                if m in best_row and pd.notnull(best_row[m]):
                    # If the combined score matches this metric, use its name
                    # Floating point comparison
                    if abs(best_row[m] - best_score) < 1e-6:
                        metric_name = m
                        break
            
            # Format date
            # run_name is usually YYYY.MM.DDHHMMSS
            # Try to parse it for display
            try:
                # Expected format: 2026.01.27153137
                # Split roughly
                date_part = run_name[:10]
                time_part = run_name[10:]
                disp_date = f"{date_part} {time_part[:2]}:{time_part[2:4]}"
            except:
                disp_date = run_name[:15]

            status = "Running" # Hard to detect if truly finished or crashed without checking logs, assuming running/done
            # We could check if 'total_gens' matches config target, but we don't have config loaded here easily.
            
            print(f"{exp_name:<40} | {disp_date:<20} | {total_gens:<5} | {best_score:<12.6f} | {metric_name:<20} | Found Solution")

        except Exception as e:
            print(f"{exp_name:<40} | {run_name:<20} | {'-':<5} | {'Error':<12} | {str(e)[:20]} | Error reading DB")

if __name__ == "__main__":
    summarize()
