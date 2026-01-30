#!/bin/bash

# List of all tasks (matching config filenames)
TASKS=(
    "circle_packing_rect_21"
    "circle_packing_square_26"
    "circle_packing_square_32"
    "first_autocorr_ineq"
    "hexagon_packing_11"
    "hexagon_packing_12"
    "minimizing_max_min_dist_14_3"
    "minimizing_max_min_dist_16_2"
    "second_autocorr_ineq"
)

# Configuration for CPU affinity
TOTAL_CORES=48
SLOT_SIZE=10
NUM_SLOTS=$((TOTAL_CORES / SLOT_SIZE)) # 4 slots: 0-9, 10-19, 20-29, 30-39

# Array to store PIDs of running jobs. Index corresponds to the core slot.
# Initialize with 0s
for ((i=0; i<NUM_SLOTS; i++)); do
    SLOT_PIDS[$i]=0
done

echo "Starting benchmark pipeline with NUM_SLOTS=$NUM_SLOTS (Cores per slot: $SLOT_SIZE)..."

get_required_slots() {
    if [[ "$1" == "minimizing_max_min_dist_14_3" ]]; then
        echo 2 # Needs 20 cores
    else
        echo 1 # Needs 10 cores
    fi
}

run_experiment() {
    local task=$1
    local variant=$2 # "qwen" or "gemini"
    local evolution_config=$3
    
    local required_slots=$(get_required_slots "$task")
    
    # Find consecutive free slots
    local assigned_start_slot=-1
    while [ $assigned_start_slot -eq -1 ]; do
        # Iterate through possible start slots
        for ((i=0; i<=NUM_SLOTS-required_slots; i++)); do
            local all_free=true
            # Check if all needed slots from i are free
            for ((j=0; j<required_slots; j++)); do
                local slot_idx=$((i + j))
                local pid=${SLOT_PIDS[$slot_idx]}
                # Check if PID is set and process is running
                if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null;
 then
                    all_free=false
                    break
                fi
            done
            
            if [ "$all_free" = true ]; then
                assigned_start_slot=$i
                break
            fi
        done

        # If no slots found, wait for any background job to finish
        if [ $assigned_start_slot -eq -1 ]; then
            wait -n
        fi
    done

    # Calculate core range
    local start_core=$((assigned_start_slot * SLOT_SIZE))
    local end_core=$((start_core + (required_slots * SLOT_SIZE) - 1))
    local core_range="${start_core}-${end_core}"

    echo "Launching task: $task ($variant) on cores $core_range (Slots $assigned_start_slot to $((assigned_start_slot + required_slots - 1)))"

    local OUTPUT_DIR="results/${task}/${variant}"
    # Ensure directory exists for the log file
    mkdir -p "$OUTPUT_DIR"
    local LOG_FILE="$OUTPUT_DIR/run.log"

    # Launch with taskset
    # Note: Hydra/Slurm config 'cpus' parameter is passed implicitly via config file, 
    # but taskset ensures affinity.
    taskset -c "$core_range" python shinka/launch_hydra.py \
        task@_global_=$task \
        evolution@_global_=$evolution_config \
        database@_global_=$task \
        output_dir=$OUTPUT_DIR \
        variant_suffix="_${variant}" > "$LOG_FILE" 2>&1 &

    local job_pid=$!
    
    # Mark all assigned slots with the new PID
    for ((j=0; j<required_slots; j++)); do
        local slot_idx=$((assigned_start_slot + j))
        SLOT_PIDS[$slot_idx]=$job_pid
    done
}

for task in "${TASKS[@]}"; do
    # Run Qwen experiment
    run_experiment "$task" "qwen" "${task}_qwen"
    
    # Run Gemini experiment
    run_experiment "$task" "gemini" "${task}_gemini"
done

# Wait for all remaining jobs
wait

echo "All tasks finished."