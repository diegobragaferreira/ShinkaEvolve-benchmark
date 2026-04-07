#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

# Load API keys and environment variables
if [ -f "export_api_keys.sh" ]; then
    echo "Loading environment variables from export_api_keys.sh..."
    source export_api_keys.sh
else
    echo "Warning: export_api_keys.sh not found. Assuming environment is already set."
fi

# List of all tasks (matching config filenames)
TASKS=(
    "circle_packing_rect_21"
    "circle_packing_square_26"
    "circle_packing_square_32"
    "hexagon_packing_11"
    "hexagon_packing_12"
    "minimizing_max_min_dist_14_3"
    "minimizing_max_min_dist_16_2"
    "first_autocorr_ineq"
    "second_autocorr_ineq"
)

EXP_NAME=${1:-"default_exp"}
NUM_ROUNDS_ARG=${2:-"1"}

# Parse NUM_ROUNDS from "num_rounds=3" or just "3"
if [[ "$NUM_ROUNDS_ARG" == num_rounds=* ]]; then
    NUM_ROUNDS=${NUM_ROUNDS_ARG#num_rounds=}
else
    NUM_ROUNDS=$NUM_ROUNDS_ARG
fi

# Configuration for CPU affinity
TOTAL_CORES=48
SLOT_SIZE=10
NUM_SLOTS=$((TOTAL_CORES / SLOT_SIZE)) # 4 slots: 0-9, 10-19, 20-29, 30-39

# Array to store PIDs of running jobs. Index corresponds to the core slot.
# Initialize with 0s
for ((i=0; i<NUM_SLOTS; i++)); do
    SLOT_PIDS[$i]=0
done

get_required_slots() {
    local task=$1
    local config_file="configs/task/${task}.yaml"
    local cpus=10 # Default fallback
    if [ -f "$config_file" ]; then
        # Extract cpus from the yaml file (e.g., 'cpus: 10')
        local extracted_cpus=$(grep "cpus:" "$config_file" | head -n 1 | awk '{print $2}')
        if [ -n "$extracted_cpus" ]; then
            cpus=$extracted_cpus
        fi
    fi
    # Calculate how many 10-core slots are needed
    echo $(( (cpus + SLOT_SIZE - 1) / SLOT_SIZE ))
}

# Build the task queue: All rounds for Qwen first, then all rounds for Gemini
PENDING_TASKS=()
for variant in "qwen"; do #"gemini"
    for ((round=1; round<=NUM_ROUNDS; round++)); do
        for task in "${TASKS[@]}"; do
            PENDING_TASKS+=("$task|$variant|${task}_${variant}|$round")
        done
    done
done

echo "Starting benchmark pipeline for experiment '$EXP_NAME' with NUM_ROUNDS=$NUM_ROUNDS..."
echo "Total experiments to run: ${#PENDING_TASKS[@]} (Slots available: $NUM_SLOTS, Cores per slot: $SLOT_SIZE)"

while [ ${#PENDING_TASKS[@]} -gt 0 ] || [ $(jobs -r | wc -l) -gt 0 ]; do
    launched_any=false
    
    # Try to launch tasks from the queue
    NEW_PENDING_TASKS=()
    for item in "${PENDING_TASKS[@]}"; do
        IFS='|' read -r task variant evolution_config round <<< "$item"
        required_slots=$(get_required_slots "$task")
        
        # Find consecutive free slots
        assigned_start_slot=-1
        for ((i=0; i<=NUM_SLOTS-required_slots; i++)); do
            all_free=true
            for ((j=0; j<required_slots; j++)); do
                slot_idx=$((i + j))
                pid=${SLOT_PIDS[$slot_idx]}
                if [ "$pid" -ne 0 ] && kill -0 "$pid" 2>/dev/null; then
                    all_free=false
                    break
                fi
            done
            
            if [ "$all_free" = true ]; then
                assigned_start_slot=$i
                break
            fi
        done

        if [ $assigned_start_slot -ne -1 ]; then
            # Calculate core range
            start_core=$((assigned_start_slot * SLOT_SIZE))
            end_core=$((start_core + (required_slots * SLOT_SIZE) - 1))
            core_range="${start_core}-${end_core}"

            echo "Launching: Task $task | Variant $variant | Round $round on cores $core_range (Slots $assigned_start_slot to $((assigned_start_slot + required_slots - 1)))"

            OUTPUT_DIR="results/${EXP_NAME}/${variant}/${round}/${task}"
            mkdir -p "$OUTPUT_DIR"
            LOG_FILE="$OUTPUT_DIR/launch_hydra.log"
            
            # Calculate a unique seed for this round for reproducibility
            SEED=$((42 + round))

            taskset -c "$core_range" python shinka/launch_hydra.py \
                task@_global_=$task \
                evolution@_global_=$evolution_config \
                database@_global_=$task \
                +evo_config.seed=$SEED \
                output_dir=$OUTPUT_DIR \
                variant_suffix="_${variant}" > "$LOG_FILE" 2>&1 &

            job_pid=$!
            for ((j=0; j<required_slots; j++)); do
                SLOT_PIDS[$((assigned_start_slot + j))]=$job_pid
            done
            launched_any=true
        else
            # Keep in queue
            NEW_PENDING_TASKS+=("$item")
        fi
    done
    PENDING_TASKS=("${NEW_PENDING_TASKS[@]}")

    if [ "$launched_any" = false ] && [ ${#PENDING_TASKS[@]} -gt 0 ]; then
        # Wait for any background job to finish before trying again
        wait -n
    elif [ ${#PENDING_TASKS[@]} -eq 0 ] && [ $(jobs -r | wc -l) -gt 0 ]; then
        # No more tasks to launch, just wait for the rest
        wait
    fi
done

echo "All benchmark experiments finished."
