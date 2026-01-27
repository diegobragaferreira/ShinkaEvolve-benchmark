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
    
    MAX_JOBS=3 # Set the maximum number of parallel experiments
    PIDS=()
    
    echo "Starting benchmark pipeline with MAX_JOBS=$MAX_JOBS..."
    
    for task in "${TASKS[@]}"; do
        # Check if we have reached the maximum number of parallel jobs
        while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
            # Wait for any job to finish
            wait -n
            
            # Remove finished PIDS from the list
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 $pid 2>/dev/null; then
                    NEW_PIDS+=($pid)
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
        done
    
        echo "Launching task: $task"
        
        # We set a fixed output directory for each task to allow resuming.
        # If the directory exists, ShinkaEvolve will automatically resume the run.
        OUTPUT_DIR="results/${task}"
        
        # We use 'hydra.run.dir' to override the default timestamp-based directory.
        # We also pass 'evolution=$task' and 'task=$task' as before.
        python shinka/launch_hydra.py \
            task@_global_=$task \
            evolution@_global_=$task \
            hydra.run.dir=$OUTPUT_DIR &
        
        # Add the new PID to the list
        PIDS+=($!)
        
        echo "------------------------------------------------"
    done
    
    # Wait for all remaining jobs to finish
    wait
    
    echo "All tasks finished."
