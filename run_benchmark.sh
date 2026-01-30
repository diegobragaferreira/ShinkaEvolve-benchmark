#!/bin/bash

# List of all tasks
TASKS=(
    "circle_packing_square_26"
    "circle_packing_square_32"
    "circle_packing_rect"
    "hexagon_packing_11"
    "hexagon_packing_12"
    "minimizing_max_min_dist_16_2"
    "minimizing_max_min_dist_14_3"
    "first_autocorr_ineq"
    "second_autocorr_ineq"
)

# Configuration for CPU affinity
TOTAL_CORES=48
CORES_PER_JOB=10
MAX_JOBS=$((TOTAL_CORES / CORES_PER_JOB))

# Array to store PIDs of running jobs. Index corresponds to the core slot.
# Initialize with 0s
for ((i=0; i<MAX_JOBS; i++)); do
    SLOT_PIDS[$i]=0
done

echo "Starting benchmark pipeline with MAX_JOBS=$MAX_JOBS (Cores per job: $CORES_PER_JOB)..."

for task in "${TASKS[@]}"; do
    # Find a free slot
    assigned_slot=-1
    while [ $assigned_slot -eq -1 ]; do
        # Check all slots to see if any are free or if the process in them has finished
        for ((i=0; i<MAX_JOBS; i++)); do
            pid=${SLOT_PIDS[$i]}
            # If pid is 0 (unused) or process is not running (kill -0 fails)
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i
                break
            fi
        done

        # If no slot is free, wait for any background job to finish before checking again
        if [ $assigned_slot -eq -1 ]; then
            wait -n
        fi
    done

    # Calculate core range for the assigned slot
    start_core=$((assigned_slot * CORES_PER_JOB))
    end_core=$((start_core + CORES_PER_JOB - 1))
    core_range="${start_core}-${end_core}"

    echo "Launching task: $task on cores $core_range (Slot $assigned_slot)"

    OUTPUT_DIR="results/${task}"

    # Launch with taskset to bind to specific cores
    taskset -c "$core_range" python shinka/launch_hydra.py \
        task@_global_=$task \
        evolution@_global_=$task \
        database@_global_=$task \
        output_dir=$OUTPUT_DIR &

    # Save the new PID to the slot
    SLOT_PIDS[$assigned_slot]=$!
    
    echo "------------------------------------------------"
done

# Wait for all remaining jobs to finish
wait

echo "All tasks finished."