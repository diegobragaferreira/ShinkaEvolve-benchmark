import json
import sys
import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

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

def parse_log_stats(run_dir):
    """Parses launch_hydra.log for duration and API errors."""
    log_path = os.path.join(run_dir, "launch_hydra.log")
    if not os.path.exists(log_path):
        return "-", "No Log"

    try:
        with open(log_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            return "-", "Empty Log"

        # Parse Time with Gap Detection
        valid_times = []
        
        # Helper to extract time from line [YYYY-MM-DD HH:MM:SS,mmm]
        def extract_time(line):
            try:
                # Assuming format [YYYY-MM-DD HH:MM:SS,mmm]
                part = line.split(']')[0]
                part = part.strip('[')
                return datetime.strptime(part, "%Y-%m-%d %H:%M:%S,%f")
            except:
                return None

        for line in lines:
            t = extract_time(line)
            if t:
                valid_times.append(t)
        
        total_seconds = 0
        if len(valid_times) > 1:
            GAP_THRESHOLD = 1800 # 30 minutes
            for i in range(1, len(valid_times)):
                diff = (valid_times[i] - valid_times[i-1]).total_seconds()
                if diff < GAP_THRESHOLD:
                    total_seconds += diff
        
        duration_str = "-"
        if total_seconds > 0:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m"
            else:
                duration_str = f"{minutes}m"

        # Check for API Errors
        api_status = "OK"
        error_keywords = {
            "RateLimit": "Rate",
            "429": "Rate",
            "Timeout": "Time",
            "ConnectError": "Conn",
            "500": "500",
            "APIError": "API",
            "ServiceUnavailable": "Serv"
        }
        
        content = "".join(lines)
        error_counts = {}
        
        for key, label in error_keywords.items():
            count = content.count(key)
            if count > 0:
                # Aggregate counts for same label (e.g. RateLimit and 429)
                error_counts[label] = error_counts.get(label, 0) + count
        
        if error_counts:
            # Sort by count desc
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            parts = []
            for label, count in sorted_errors:
                parts.append(f"{label}:{count}")
            
            api_status = ",".join(parts)
            if len(api_status) > 18:
                 api_status = api_status[:17] + "+"

        return duration_str, api_status

    except Exception:
        return "-", "Log Err"

def process_and_print_run(run_path, display_name):
    """Helper to process a single run directory and print its status line."""
    db_path = os.path.join(run_path, "evolution_db.sqlite")
    if not os.path.exists(db_path):
        db_path = os.path.join(run_path, "evolution.db")
    
    if not os.path.exists(db_path):
        print(f"{display_name:<37} | {'N/A':<10} | {'-':<4} | {'-':<12} | {'-':<17} | {'-':<8} | {'-':<10} | No DB")
        return

    run_name = os.path.basename(run_path)
    
    # Get Log Stats
    duration, api_status = parse_log_stats(run_path)

    try:
        df = load_programs_to_df(db_path)
        if df is None or df.empty:
            print(f"{display_name:<37} | {run_name[:10]:<10} | {'0':<4} | {'-':<12} | {'-':<17} | {duration:<8} | {api_status:<10} | Empty")
            return
        
        if 'correct' in df.columns:
            correct_df = df[df['correct'] == True]
        else:
            correct_df = df

        total_gens = (df['generation'].max() + 1) if 'generation' in df.columns else 0
        
        if correct_df.empty:
            print(f"{display_name:<37} | {run_name[:10]:<10} | {total_gens:<4} | {'-':<12} | {'-':<17} | {duration:<8} | {api_status:<10} | No valid")
            return

        # Best score
        best_idx = correct_df['combined_score'].idxmax()
        best_row = correct_df.loc[best_idx]
        best_score = best_row['combined_score']
        
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
        
        print(f"{display_name:<37} | {run_name[:10]:<10} | {total_gens:<4} | {best_score:<12.6f} | {derived_val_str:<17} | {duration:<8} | {api_status:<18} | OK")

    except Exception as e:
        print(f"{display_name:<37} | {run_name[:10]:<10} | {'-':<4} | {'Error':<12} | {'-':<17} | {duration:<8} | {api_status:<18} | Error")

def generate_report(exp_name=None, round_num=None):
    if load_programs_to_df is None:
        print("Error: Could not import 'shinka.utils.load_df'. Make sure the environment is active and shinka is installed.")
        return

    base_results_dir = RESULTS_DIR
    if exp_name:
        base_results_dir = os.path.join(RESULTS_DIR, exp_name)

    if not os.path.exists(base_results_dir):
        print(f"Results directory '{base_results_dir}' not found.")
        return

    # Find all database files recursively
    results = []
    for root, dirs, files in os.walk(base_results_dir):
        if "evolution.db" in files or "evolution_db.sqlite" in files:
            results.append(root)

    if not results:
        print(f"No results found in {base_results_dir}")
        return

    # Try to extract info from path
    # Supported structures:
    # 1. results/task/variant/run
    # 2. results/exp/task/variant/round
    # 3. results/exp/variant/round/task
    
    parsed_results = []
    for r in results:
        rel_path = os.path.relpath(r, RESULTS_DIR)
        parts = rel_path.split(os.sep)
        
        # Default values
        res_exp = "Root"
        res_task = "Unknown"
        res_variant = "default"
        res_round = "1"
        
        if len(parts) >= 4:
            # Check for results/exp/variant/round/task
            if parts[1] in ["qwen", "gemini"]:
                res_exp = parts[0]
                res_variant = parts[1]
                res_round = parts[2]
                res_task = parts[3]
            # Check for results/exp/task/variant/round
            elif parts[2] in ["qwen", "gemini"]:
                res_exp = parts[0]
                res_task = parts[1]
                res_variant = parts[2]
                res_round = parts[3]
        elif len(parts) == 3:
            # results/task/variant/run
            res_task = parts[0]
            res_variant = parts[1]
            res_round = parts[2]
        elif len(parts) == 2:
            # results/task/variant (direct files)
            res_task = parts[0]
            res_variant = parts[1]

        # Filter by round if requested
        if round_num and res_round != str(round_num):
            continue
            
        parsed_results.append({
            "path": r,
            "exp": res_exp,
            "task": res_task,
            "variant": res_variant,
            "round": res_round
        })

    # Sort and group
    parsed_results.sort(key=lambda x: (x["exp"], x["task"], x["variant"], x["round"]))

    print(f"Report for Experiment: {exp_name if exp_name else 'All'} | Round: {round_num if round_num else 'All'}")
    print(f"{'Exp':<12} | {'Task Name (Variant)':<37} | {'Rnd':<4} | {'Gen':<4} | {'Best Score':<12} | {'Const':<17} | {'Time':<8} | {'API':<18} | {'Status'}")
    print("-" * 171)

    summary_data = []
    for res in parsed_results:
        display_name = f"{res['task']} ({res['variant']})"
        
        db_path = os.path.join(res["path"], "evolution_db.sqlite")
        if not os.path.exists(db_path):
            db_path = os.path.join(res["path"], "evolution.db")
        
        duration, api_status = parse_log_stats(res["path"])
        
        try:
            df = load_programs_to_df(db_path)
            if df is None or df.empty:
                print(f"{res['exp']:<12} | {display_name:<37} | {res['round']:<4} | {'0':<4} | {'-':<12} | {'-':<17} | {duration:<8} | {api_status:<18} | Empty")
                continue
            
            if 'correct' in df.columns:
                correct_df = df[df['correct'] == True]
            else:
                correct_df = df

            total_gens = (df['generation'].max() + 1) if 'generation' in df.columns else 0
            
            if correct_df.empty:
                print(f"{res['exp']:<12} | {display_name:<37} | {res['round']:<4} | {total_gens:<4} | {'-':<12} | {'-':<17} | {duration:<8} | {api_status:<18} | No valid")
                continue

            best_idx = correct_df['combined_score'].idxmax()
            best_row = correct_df.loc[best_idx]
            best_score = best_row['combined_score']
            
            derived_val_str = "-"
            derived_val = None
            prefix = ""
            if "first_autocorr" in res['task']:
                val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
                derived_val_str = f"c1={val:.6f} ↓"
                derived_val = val
                prefix = "c1"
            elif "second_autocorr" in res['task']:
                derived_val_str = f"c2={best_score:.6f} ↑"
                derived_val = best_score
                prefix = "c2"
            elif "third_autocorr" in res['task']:
                val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
                derived_val_str = f"c3={val:.6f} ↓"
                derived_val = val
                prefix = "c3"
            elif "hexagon_packing" in res['task']:
                val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
                derived_val_str = f"s={val:.6f} ↓"
                derived_val = val
                prefix = "s"
            elif "minimizing_max_min_dist" in res['task']:
                val = 1.0 / best_score if abs(best_score) > 1e-9 else 0
                derived_val_str = f"r={val:.6f} ↓"
                derived_val = val
                prefix = "r"
            elif "circle_packing" in res['task']:
                derived_val_str = f"sum={best_score:.6f} ↑"
                derived_val = best_score
                prefix = "sum"
            
            print(f"{res['exp']:<12} | {display_name:<37} | {res['round']:<4} | {total_gens:<4} | {best_score:<12.6f} | {derived_val_str:<17} | {duration:<8} | {api_status:<18} | OK")

            summary_data.append({
                "task": res['task'],
                "variant": res['variant'],
                "round": res['round'],
                "score": best_score,
                "const": derived_val,
                "prefix": prefix
            })

        except Exception as e:
            print(f"{res['exp']:<12} | {display_name:<37} | {res['round']:<4} | {'-':<4} | {'Error':<12} | {'-':<17} | {duration:<8} | {api_status:<18} | Error")

    if summary_data:
        print("\n" + "="*95)
        print(f"{'Summary per Task (Variant)':<45} | {'Best Const':<30} | {'Mean Const':<12}")
        print("-" * 95)
        
        df_summary = pd.DataFrame(summary_data)
        for (task, variant), group in df_summary.groupby(['task', 'variant']):
            display_name = f"{task} ({variant})"
            best_idx = group['score'].idxmax()
            best_row = group.loc[best_idx]
            
            # Use constant values if available
            if best_row['const'] is not None:
                mean_const = group['const'].mean()
                prefix = best_row['prefix']
                best_val_str = f"{prefix:>3} = {best_row['const']:<10.6f} (R{best_row['round']})"
                print(f"{display_name:<45} | {best_val_str:<30} | {mean_const:<12.6f}")
            else:
                mean_score = group['score'].mean()
                best_val_str = f"{best_row['score']:<10.6f} (R{best_row['round']})"
                print(f"{display_name:<45} | {best_val_str:<30} | {mean_score:<12.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: manage_benchmark_status.py [check|mark|report] [task_name/exp_name] [round]")
        print("  check  : Check if task is completed (exit 0=yes, 1=no)")
        print("  mark   : Mark task as completed")
        print("  report : Show summary of all results")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "report":
        exp = sys.argv[2] if len(sys.argv) > 2 else None
        rnd = sys.argv[3] if len(sys.argv) > 3 else None
        generate_report(exp, rnd)
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
