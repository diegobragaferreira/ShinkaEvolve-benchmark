import json
import sys
import os

STATUS_FILE = "benchmark_status.json"

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: manage_benchmark_status.py [check|mark] [task_name]")
        sys.exit(1)
        
    command = sys.argv[1]
    task = sys.argv[2]
    
    if command == "check":
        if is_completed(task):
            sys.exit(0) # Success means completed
        else:
            sys.exit(1) # Failure means not completed
    elif command == "mark":
        mark_completed(task)
        sys.exit(0)
