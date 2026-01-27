#!/bin/bash

# List of all tasks
TASKS=(
    "rect_circle_packing"
    "first_autocorr_ineq"
    "second_autocorr_ineq"
    "third_autocorr_ineq"
    "heilbronn_convex_13"
    "heilbronn_convex_14"
    "heilbronn_triangle"
    "kissing_number"
    "minimizing_max_min_dist_2"
    "minimizing_max_min_dist_3"
    "circle_packing_square_26"
    "circle_packing_square_32"
    "circle_packing_square_general"
    "hexagon_packing_11"
    "hexagon_packing_12"
)

echo "Starting benchmark pipeline..."

for task in "${TASKS[@]}"; do
    # Check if task is already completed
    python manage_benchmark_status.py check "$task"
    if [ $? -eq 0 ]; then
        echo "Skipping task: $task (already completed)"
        continue
    fi

    echo "Launching task: $task"
    # Note: Using @_global_ suffix to correctly override the defaults in configs/config.yaml
    python shinka/launch_hydra.py task@_global_=$task evolution@_global_=$task
    
    if [ $? -eq 0 ]; then
        echo "Task $task completed successfully."
        python manage_benchmark_status.py mark "$task"
    else
        echo "Task $task failed."
        # Optionally exit or continue on failure. Continuing for now.
    fi
    echo "------------------------------------------------"
done

echo "All tasks launched."