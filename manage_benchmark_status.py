import json
import sys
import os
import glob
import pandas as pd
from pathlib import Path

# Ensure shinka is in python path to allow importing utils
sys.path.append(os.getcwd())

try:
    from shinka.utils.load_df import load_programs_to_df
except ImportError:
    # Fallback if shinka not installed/found, though needed for report
    load_programs_to_df = None

STATUS_FILE = "benchmark_status.json"
RESULTS_DIR = "results"

def load_status():
    if not os.path.exists(STATUS_FILE):
        return {}
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_status(status):
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)

def is_completed(task_name):
    status = load_status()
    return status.get(task_name) == "success"

def mark_completed(task_name):
    status = load_status()
    status[task_name] = "success"
    save_status(status)

def get_latest_run_dir(exp_dir):
    """Finds the latest run directory within an experiment directory."""
    # Check for direct database file (newer/current format seems to be .sqlite)
    if os.path.exists(os.path.join(exp_dir, "evolution_db.sqlite")):
        return exp_dir
    # Check for legacy .db file
    if os.path.exists(os.path.join(exp_dir, "evolution.db")):
        return exp_dir

    run_dirs = [d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)]
    if not run_dirs:
        return None
    
    # Sort by name (timestamp)
    run_dirs.sort(key=lambda x: os.path.basename(x))
    return run_dirs[-1]

def process_and_print_run(run_path, display_name):
    """Helper to process a single run directory and print its status line."""
    db_path = os.path.join(run_path, "evolution_db.sqlite")
    if not os.path.exists(db_path):
        db_path = os.path.join(run_path, "evolution.db")
    
    if not os.path.exists(db_path):
        print(f"{display_name:<37} | {'N/A':<10} | {'-':<4} | {'-':<12} | {'-':<25} | {'-':<17} | No DB")
        return

    run_name = os.path.basename(run_path)
    # If run_path is the variant dir itself, the name might be 'qwen' or 'gemini'
    # We might want to use the parent dir's name if we need a date, 
    # but the script often puts runs in timestamped folders.
    
    try:
        df = load_programs_to_df(db_path)
        if df is None or df.empty:
            print(f"{display_name:<37} | {run_name[:10]:<10} | {'0':<4} | {'-':<12} | {'-':<25} | {'-':<17} | Empty")
            return
        
        if 'correct' in df.columns:
            correct_df = df[df['correct'] == True]
        else:
            correct_df = df

        total_gens = (df['generation'].max() + 1) if 'generation' in df.columns else 0
        
        if correct_df.empty:
            print(f"{display_name:<37} | {run_name[:10]:<10} | {total_gens:<4} | {'-':<12} | {'-':<25} | {'-':<17} | No valid")
            return

        # Best score
        best_idx = correct_df['combined_score'].idxmax()
        best_row = correct_df.loc[best_idx]
        best_score = best_row['combined_score']
        
        # Identify metric
        metric_name = "combined_score"
        priority_metrics = ['benchmark_ratio', 'avg_benchmark_ratio', 'sum_radii', 'radii_sum', 'inv_c1', 'inv_c3', 'inv_outer_hex_side_length', 'c2']
        for m in priority_metrics:
            if m in best_row and pd.notnull(best_row[m]):
                try:
                    if abs(float(best_row[m]) - float(best_score)) < 1e-6:
                        metric_name = m
                        break
                except:
                    pass
        
        # Special handling to display constants
        derived_val_str = "-"
        if "first_autocorr" in display_name:
            val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
            derived_val_str = f"c1={val:.6f} ↓"
        elif "second_autocorr" in display_name:
            derived_val_str = f"c2={best_score:.6f} ↑"
        elif "third_autocorr" in display_name:
            val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
            derived_val_str = f"c3={val:.6f} ↓"
        elif "hexagon_packing" in display_name:
            val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
            derived_val_str = f"s={val:.6f} ↓"
        elif "minimizing_max_min_dist" in display_name:
            val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
            derived_val_str = f"r={val:.6f} ↓"
        elif "circle_packing" in display_name:
            derived_val_str = f"sum={best_score:.6f} ↑"
        
        print(f"{display_name:<37} | {run_name[:10]:<10} | {total_gens:<4} | {best_score:<12.6f} | {metric_name:<25} | {derived_val_str:<17} | OK")

    except Exception as e:
        print(f"{display_name:<37} | {run_name[:10]:<10} | {'-':<4} | {'Error':<12} | {str(e)[:25]:<25} | {'-':<17} | Error")

def generate_report():
    if load_programs_to_df is None:
        print("Error: Could not import 'shinka.utils.load_df'. Make sure the environment is active and shinka is installed.")
        return

    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found.")
        return

    task_dirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    task_dirs.sort()

    print(f"{'Task Name (Variant)':<37} | {'Run Dir':<10} | {'Gen':<4} | {'Best Score':<12} | {'Metric':<25} | {'Const':<17} | {'Status'}")
    print("-" * 141)

    for task_path in task_dirs:
        task_name = os.path.basename(task_path)
        
        if task_name.startswith("__") or task_name.startswith("."):
            continue

        # Check for sub-variants (qwen/gemini)
        variants = [d for d in glob.glob(os.path.join(task_path, "*")) if os.path.isdir(d)]
        variants.sort()
        
        has_named_variants = False
        for var_path in variants:
            var_name = os.path.basename(var_path)
            if var_name in ["qwen", "gemini"]:
                has_named_variants = True
                latest_run = get_latest_run_dir(var_path)
                if latest_run:
                    process_and_print_run(latest_run, f"{task_name} ({var_name})")
                else:
                    print(f"{task_name + ' (' + var_name + ')':<37} | {'N/A':<10} | {'-':<4} | {'-':<12} | {'-':<25} | {'-':<17} | No runs")

        if not has_named_variants:
            # Fallback for old structure or non-variant tasks
            latest_run = get_latest_run_dir(task_path)
            if latest_run:
                process_and_print_run(latest_run, task_name)
            else:
                print(f"{task_name:<37} | {'N/A':<10} | {'-':<4} | {'-':<12} | {'-':<25} | {'-':<17} | No runs")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: manage_benchmark_status.py [check|mark|report] [task_name]")
        print("  check  : Check if task is completed (exit 0=yes, 1=no)")
        print("  mark   : Mark task as completed")
        print("  report : Show summary of all results")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "report":
        generate_report()
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print(f"Usage: manage_benchmark_status.py {command} [task_name]")
        sys.exit(1)

    task = sys.argv[2]
    
    if command == "check":
        if is_completed(task):
            sys.exit(0)
        else:
            sys.exit(1)
    elif command == "mark":
        mark_completed(task)
        sys.exit(0)
