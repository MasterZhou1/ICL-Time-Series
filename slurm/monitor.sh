#!/bin/bash
# ICL Time Series - SLURM Monitoring
# Clean, minimal monitoring for experiment progress

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Utility functions
log() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}" >&2; }

show_help() {
    cat << EOF
ICL Time Series - SLURM Monitoring

USAGE:
    $0 [OPTIONS] [JOB_IDS...]

OPTIONS:
    --live, -l     Live dashboard (default)
    --quick, -q    Quick status
    --multi, -m    Multi-job monitoring
    --refresh N    Refresh interval (default: 30s)
    --help         Show this help

EXAMPLES:
    $0                    # Live dashboard
    $0 --quick           # Quick status
    $0 --multi 123 456   # Monitor specific jobs
EOF
}

# Get ICL jobs
get_icl_jobs() {
    squeue -u "$USER" -o "%.18i %.12j %.8T %.10M %.15R" --noheader 2>/dev/null | \
    grep -E "(icl_|ICL)" || echo ""
}

# Get detailed job info via scontrol (single-line output)
get_job_detail() {
    local job_id="$1"
    scontrol show job -o "$job_id" 2>/dev/null || echo ""
}

# Parse fields from scontrol job line (Key=Value ...)
parse_job_field() {
    local line="$1" key="$2"
    echo "$line" | tr ' ' '\n' | grep -E "^${key}=" | sed -E "s/^${key}=//"
}

# Parse requested GPU type from job name pattern icl_<gpuType>
parse_gpu_request_from_name() {
    local name="$1"
    if [[ "$name" =~ ^icl_([A-Za-z0-9_]+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown"
    fi
}

# Derive GPU type and count from Gres/TRES
parse_gpu_info() {
    local line="$1"
    local gres tres gpu_type gpu_count
    gres=$(parse_job_field "$line" "Gres")
    tres=$(parse_job_field "$line" "TRES")
    # Default
    gpu_type="unknown"
    gpu_count="0"
    if [[ -n "$gres" ]]; then
        # Examples: gpu:a5000:1 or (null)
        if echo "$gres" | grep -q "gpu:"; then
            gpu_type=$(echo "$gres" | sed -E 's/.*gpu:([^:]+).*/\1/')
            gpu_count=$(echo "$gres" | sed -E 's/.*gpu:[^:]+:([0-9]+).*/\1/')
        fi
    fi
    if [[ "$gpu_count" == "0" && -n "$tres" ]]; then
        # TRES=cpu=8,mem=32768M,node=1,billing=8,gres/gpu=1
        gpu_count=$(echo "$tres" | sed -n 's/.*gres\/gpu=\([0-9]\+\).*/\1/p')
    fi
    # Fallback to 1 when we requested a GPU but count is unknown
    if [[ "$gpu_type" != "unknown" && ( -z "$gpu_count" || "$gpu_count" == "0" ) ]]; then
        gpu_count=1
    fi
    echo "$gpu_type $gpu_count"
}

# Parse total models from job name: icl_<gpuType>_<start>_<end>
parse_total_from_name() {
    local name="$1"
    if [[ "$name" =~ ^icl_[A-Za-z0-9_]+_([0-9]+)_([0-9]+)$ ]]; then
        local s=${BASH_REMATCH[1]}
        local e=${BASH_REMATCH[2]}
        echo $(( e - s ))
    else
        echo 0
    fi
}

# Compute ETA and GPU-hours used
compute_runtime_stats() {
    local line="$1"
    local start time_limit state run_time
    start=$(parse_job_field "$line" "StartTime")
    time_limit=$(parse_job_field "$line" "TimeLimit")
    state=$(parse_job_field "$line" "JobState")
    run_time=$(parse_job_field "$line" "RunTime")

    # Normalize timestamps to epoch
    local now_epoch=$(date +%s)
    local start_epoch=0
    if [[ -n "$start" && "$start" != "Unknown" && "$start" != "N/A" ]]; then
        start_epoch=$(date -d "$start" +%s 2>/dev/null || echo 0)
    fi

    # Convert TimeLimit (DD-HH:MM:SS or HH:MM:SS) to seconds
    local tl_sec=0
    if [[ -n "$time_limit" ]]; then
        if echo "$time_limit" | grep -q "-"; then
            # DD-HH:MM:SS
            local dd=$(echo "$time_limit" | cut -d- -f1)
            local hms=$(echo "$time_limit" | cut -d- -f2)
            local hh=$(echo "$hms" | cut -d: -f1)
            local mm=$(echo "$hms" | cut -d: -f2)
            local ss=$(echo "$hms" | cut -d: -f3)
            tl_sec=$((10#$dd*86400 + 10#$hh*3600 + 10#$mm*60 + 10#$ss))
        else
            # HH:MM:SS
            local hh=$(echo "$time_limit" | cut -d: -f1)
            local mm=$(echo "$time_limit" | cut -d: -f2)
            local ss=$(echo "$time_limit" | cut -d: -f3)
            tl_sec=$((10#$hh*3600 + 10#$mm*60 + 10#$ss))
        fi
    fi

    local elapsed=0
    if [[ -n "$run_time" ]]; then
        # RunTime format HH:MM:SS or DD-HH:MM:SS
        if echo "$run_time" | grep -q "-"; then
            local dd=$(echo "$run_time" | cut -d- -f1)
            local hms=$(echo "$run_time" | cut -d- -f2)
            local hh=$(echo "$hms" | cut -d: -f1)
            local mm=$(echo "$hms" | cut -d: -f2)
            local ss=$(echo "$hms" | cut -d: -f3)
            elapsed=$((10#$dd*86400 + 10#$hh*3600 + 10#$mm*60 + 10#$ss))
        else
            local hh=$(echo "$run_time" | cut -d: -f1)
            local mm=$(echo "$run_time" | cut -d: -f2)
            local ss=$(echo "$run_time" | cut -d: -f3)
            elapsed=$((10#$hh*3600 + 10#$mm*60 + 10#$ss))
        fi
    elif [[ $start_epoch -gt 0 ]]; then
        elapsed=$((now_epoch - start_epoch))
    fi

    local eta_secs=$(( tl_sec > 0 ? (tl_sec - elapsed) : 0 ))
    (( eta_secs < 0 )) && eta_secs=0
    local finish_epoch=$(( now_epoch + eta_secs ))

    echo "$elapsed $eta_secs $finish_epoch"
}

# ETA based on training progress in logs (per-config average)
estimate_eta_from_progress() {
    local line="$1" total="$2" completed="$3"
    # Get elapsed seconds from runtime stats
    local stats=$(compute_runtime_stats "$line")
    local elapsed=$(echo "$stats" | awk '{print $1}')
    local remaining=$(( total > completed ? (total - completed) : 0 ))
    local eta_secs=0
    if (( completed > 0 )); then
        # Average seconds per completed config times remaining
        eta_secs=$(awk -v e="$elapsed" -v c="$completed" -v r="$remaining" 'BEGIN{ if (c>0){ printf "%.0f", (e/c)*r } else { print 0 } }')
    else
        eta_secs=0
    fi
    echo "$eta_secs"
}

# Health check based on logs and errors
job_health() {
    local job_id="$1"
    local out_log=$(find "$LOGS_DIR" -name "*${job_id}*.out" 2>/dev/null | head -1)
    local err_log=$(find "$LOGS_DIR" -name "*${job_id}*.err" 2>/dev/null | head -1)

    # Default OK
    local status="OK"
    local details=""

    # Check for Python traceback
    if [[ -f "$err_log" ]] && grep -q "Traceback" "$err_log"; then
        status="ERROR"
        details="Traceback in err log"
    fi

    # Stale output check (no updates in last 10 minutes)
    if [[ -f "$out_log" ]]; then
        local now=$(date +%s)
        local mtime=$(date -r "$out_log" +%s 2>/dev/null || echo "$now")
        local delta=$(( now - mtime ))
        if (( delta > 600 )) && [[ "$status" == "OK" ]]; then
            status="WARN"
            details="No log updates >10m"
        fi
    fi

    echo "$status" "$details"
}

# Extract current activity from log
get_current_activity() {
    local job_id="$1"
    local out_log=$(find "$LOGS_DIR" -name "*${job_id}*.out" 2>/dev/null | head -1)
    if [[ -f "$out_log" ]]; then
        local line=$(grep -E "Processing config|✅ Success:|❌ Failed:|Running [0-9]+ configurations" "$out_log" | tail -1)
        echo "$line"
    else
        echo ""
    fi
}

# Compute throughput in configs/hour
compute_throughput() {
    local elapsed="$1" completed="$2"
    if (( elapsed > 0 && completed > 0 )); then
        awk -v e=$elapsed -v c=$completed 'BEGIN{ printf "%.2f", (c*3600.0)/e }'
    else
        echo "0.00"
    fi
}

# Get experiment progress
get_progress() {
    # Determine checkpoint directories from YAML (fallback to defaults)
    local dirs
    dirs=$(python - <<'PY'
import os
from pathlib import Path
try:
    from utils.config import load_config
    project_root = Path(os.environ.get('PROJECT_ROOT', '.'))
    cfg_cs = load_config(project_root / 'configs' / 'context_scaling.yaml')
    cfg_ls = load_config(project_root / 'configs' / 'lsa_layers.yaml')
    print(cfg_cs.get('output', {}).get('checkpoints_dir', 'context_scaling/checkpoints'))
    print(cfg_ls.get('output', {}).get('checkpoints_dir', 'lsa_layers/checkpoints'))
except Exception:
    print('context_scaling/checkpoints')
    print('lsa_layers/checkpoints')
PY
)
    local context_rel=$(echo "$dirs" | sed -n '1p')
    local lsa_rel=$(echo "$dirs" | sed -n '2p')
    local context_models=$(find "$EXPERIMENTS_DIR/$context_rel" -name "best_model.pt" 2>/dev/null | wc -l)
    local lsa_models=$(find "$EXPERIMENTS_DIR/$lsa_rel" -name "best_model.pt" 2>/dev/null | wc -l)
    echo "$context_models $lsa_models $((context_models + lsa_models))"
}

# Get job progress from log
get_job_progress() {
    local job_id="$1"
    local log_file=$(find "$LOGS_DIR" -name "*${job_id}*.out" 2>/dev/null | head -1)
    
    if [[ ! -f "$log_file" ]]; then
        echo "0 0 0"
        return
    fi
    
    # Primary extraction from log
    local total
    total=$(grep -o "Running [0-9]* configurations" "$log_file" | head -1 | grep -o "[0-9]*" || true)
    local completed
    completed=$(grep -o "Progress: [0-9]*/[0-9]* completed" "$log_file" | tail -1 | grep -o "[0-9]*" | head -1 || true)
    local successful
    successful=$(grep -c "✅ Success:" "$log_file" 2>/dev/null || echo "0")

    # Sanitize to numeric-only, default 0
    total=${total//[^0-9]/}
    completed=${completed//[^0-9]/}
    successful=${successful//[^0-9]/}
    : "${total:=0}"; : "${completed:=0}"; : "${successful:=0}"

    # Fallback: parse from job name if total is unknown
    if (( total == 0 )); then
        local name=$(squeue -j "$job_id" -o "%j" --noheader 2>/dev/null || echo "")
        local t_from_name=$(parse_total_from_name "$name")
        if (( t_from_name > 0 )); then total=$t_from_name; fi
    fi

    # Fallback: infer from current activity "Processing config K/T"
    if (( total == 0 || completed == 0 )); then
        local proc_line
        proc_line=$(grep -o "Processing config [0-9]*/[0-9]*" "$log_file" | tail -1 || true)
        if [[ -n "$proc_line" ]]; then
            local k t
            k=$(echo "$proc_line" | sed -E 's/.*Processing config ([0-9]+)\/[0-9]+.*/\1/')
            t=$(echo "$proc_line" | sed -E 's/.*Processing config [0-9]+\/([0-9]+).*/\1/')
            k=${k//[^0-9]/}; t=${t//[^0-9]/}
            if (( total == 0 && t > 0 )); then total=$t; fi
            # If we have started processing, treat completed as k-1 minimally
            if (( completed == 0 && k > 1 )); then completed=$((k-1)); fi
        fi
    fi

    # Fallback: use number of successful configs if no completed yet
    if (( completed == 0 && successful > 0 )); then
        completed=$successful
    fi

    echo "$total $completed $successful"
}

# Draw progress bar
progress_bar() {
    local current="$1" total="$2" width="${3:-20}"
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "["
    printf "%*s" $filled | tr ' ' '='
    [[ $filled -lt $width ]] && printf ">"
    printf "%*s" $empty | tr ' ' '-'
    printf "] %d/%d" $current $total
}

# Quick status
show_quick() {
    echo "ICL Experiments Status"
    echo "====================="
    
    # Job status
    local jobs=$(get_icl_jobs)
    if [[ -n "$jobs" ]]; then
        echo "Running Jobs:"
        echo "$jobs" | while read -r job_id name state runtime node; do
            local line=$(get_job_detail "$job_id")
            local gpu_info=$(parse_gpu_info "$line")
            local gpu_type=$(echo "$gpu_info" | awk '{print $1}')
            local gpu_count=$(echo "$gpu_info" | awk '{print $2}')
            # Fallback to requested type from name
            if [[ "$gpu_type" == "unknown" || -z "$gpu_type" ]]; then
                gpu_type=$(parse_gpu_request_from_name "$name")
            fi
            local stats=$(compute_runtime_stats "$line")
            local elapsed=$(echo "$stats" | awk '{print $1}')
            # Progress-based ETA
            local progress=($(get_job_progress "$job_id"))
            local total_p="${progress[0]}" completed_p="${progress[1]}"
    local eta=0
    if (( total_p > 0 )); then
        eta=$(estimate_eta_from_progress "$line" "$total_p" "$completed_p")
    fi
    local now_ts=$(date +%s)
    local finish=$now_ts
    if (( eta > 0 )); then finish=$(( now_ts + eta )); fi
    local finish_str=$(date -d @${finish} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "n/a")
            local health=( $(job_health "$job_id") )
            local hstat=${health[0]}
            local hmsg=${health[@]:1}
            # GPU-hours used approx
            local gpu_hours=$(awk -v g=$gpu_count -v e=$elapsed 'BEGIN{printf "%.2f", (g*e)/3600.0}')
            local act=$(get_current_activity "$job_id")
            # GPU memory now (if available)
            local gpu_mem=""
            if command -v nvidia-smi >/dev/null 2>&1; then
                local ml=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 2>/dev/null || true)
                if [[ -n "$ml" ]]; then
                    local used=$(echo "$ml" | awk -F, '{gsub(/^ +| +$/,"",$1); print $1}')
                    local tot=$(echo "$ml" | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')
                    gpu_mem=" | GPU_MEM=${used}/${tot} MiB"
                fi
            fi
            local tput=$(compute_throughput "$elapsed" "$completed_p")
            local eta_str="--"
            if (( eta > 0 )); then eta_str=$(printf "%02dh%02dm" $((eta/3600)) $(((eta%3600)/60))); fi
            echo "  Job $job_id ($name) - $state runtime=$runtime | GPU=${gpu_type}x${gpu_count}${gpu_mem} used~${gpu_hours} GPUh | Progress=${completed_p}/${total_p} (~${tput} cfg/h) | ETA~${eta_str} finish=$finish_str | Health=$hstat ${hmsg} | ${act}"
        done
    else
        echo "No running ICL jobs"
    fi
    
    # Progress
    local progress=($(get_progress))
    echo ""
    echo "Progress:"
    echo "  Context Scaling: ${progress[0]}/$CONTEXT_TOTAL"
    echo "  LSA Layers: ${progress[1]}/$LSA_TOTAL"
    echo "  Overall: ${progress[2]}/$TOTAL_MODELS"
}

# Live dashboard
show_live() {
    local refresh="${REFRESH_INTERVAL:-30}"
    
    # Clear screen and hide cursor
    clear
    echo -e "\033[?25l"
    
    # Cleanup
    trap 'echo -e "\033[?25h"; exit 0' INT TERM
    
    while true; do
        echo -e "\033[H\033[2J"
        echo "ICL Experiments Live Monitor - $(date '+%H:%M:%S')"
        echo "================================================"
        
        # Job status
        local jobs=$(get_icl_jobs)
        if [[ -n "$jobs" ]]; then
            echo "Running Jobs:"
            echo "$jobs" | while read -r job_id name state runtime node; do
                local progress=($(get_job_progress "$job_id"))
                local total="${progress[0]}" completed="${progress[1]}" successful="${progress[2]}"
                local line=$(get_job_detail "$job_id")
                local gpu_info=$(parse_gpu_info "$line")
                local gpu_type=$(echo "$gpu_info" | awk '{print $1}')
                local gpu_count=$(echo "$gpu_info" | awk '{print $2}')
                if [[ "$gpu_type" == "unknown" || -z "$gpu_type" ]]; then
                    gpu_type=$(parse_gpu_request_from_name "$name")
                fi
                local stats=$(compute_runtime_stats "$line")
                local elapsed=$(echo "$stats" | awk '{print $1}')
            local eta=0
            if (( total > 0 )); then
                eta=$(estimate_eta_from_progress "$line" "$total" "$completed")
            fi
            local now_ts=$(date +%s)
            local finish=$now_ts
            if (( eta > 0 )); then finish=$(( now_ts + eta )); fi
                local finish_str=$(date -d @${finish} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "n/a")
                local health=( $(job_health "$job_id") )
                local hstat=${health[0]}
                local hmsg=${health[@]:1}
                local gpu_hours=$(awk -v g=$gpu_count -v e=$elapsed 'BEGIN{printf "%.2f", (g*e)/3600.0}')
                local act=$(get_current_activity "$job_id")
                local tput=$(compute_throughput "$elapsed" "$completed")
                local eta_str="--"; if (( eta > 0 )); then eta_str=$(printf "%02dh%02dm" $((eta/3600)) $(((eta%3600)/60))); fi
                echo -n "  Job $job_id [$name | ${gpu_type}x${gpu_count} | used~${gpu_hours} GPUh | Progress=${completed}/${total} (~${tput} cfg/h) | ETA~${eta_str} finish=$finish_str | Health=$hstat ${hmsg} | ${act}] "
                if [[ $total -gt 0 ]]; then
                    progress_bar $completed $total
                    echo " ($successful successful)"
                else
                    echo "Starting..."
                fi
            done
        else
            echo "No running jobs"
        fi
        
        # Overall progress
        echo ""
        echo "Overall Progress:"
        local progress=($(get_progress))
        echo -n "  Context Scaling: "
        progress_bar ${progress[0]} $CONTEXT_TOTAL
        echo ""
        echo -n "  LSA Layers:      "
        progress_bar ${progress[1]} $LSA_TOTAL
        echo ""
        echo -n "  Total:           "
        progress_bar ${progress[2]} $TOTAL_MODELS
        echo ""
        
        echo ""
        echo "Press Ctrl+C to exit | Refresh: ${refresh}s"
        
        sleep $refresh
    done
}

# Multi-job monitoring
show_multi() {
    local job_ids=("$@")
    local refresh="${REFRESH_INTERVAL:-30}"
    
    while true; do
        clear
        echo "ICL Multi-Job Monitor - $(date '+%H:%M:%S')"
        echo "=========================================="
        
        for job_id in "${job_ids[@]}"; do
            if [[ -z "$job_id" ]]; then continue; fi
            
            local job_info=$(squeue -j "$job_id" -o "%.8i %.12j %.10T %.10M %.15R" --noheader 2>/dev/null || echo "")
            
            if [[ -n "$job_info" ]]; then
                local name state runtime node
                read -r _ name state runtime node <<< "$job_info"

                local progress=($(get_job_progress "$job_id"))
                local total="${progress[0]}" completed="${progress[1]}" successful="${progress[2]}"
                local line=$(get_job_detail "$job_id")
                local gpu_info=$(parse_gpu_info "$line")
                local gpu_type=$(echo "$gpu_info" | awk '{print $1}')
                local gpu_count=$(echo "$gpu_info" | awk '{print $2}')
                if [[ "$gpu_type" == "unknown" || -z "$gpu_type" ]]; then
                    gpu_type=$(parse_gpu_request_from_name "$name")
                fi
                local stats=$(compute_runtime_stats "$line")
                local elapsed=$(echo "$stats" | awk '{print $1}')
                local eta=0
                if (( total > 0 )); then
                    eta=$(estimate_eta_from_progress "$line" "$total" "$completed")
                fi
                local now_ts=$(date +%s)
                local finish=$now_ts
                if (( eta > 0 )); then finish=$(( now_ts + eta )); fi
                local finish_str=$(date -d @${finish} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "n/a")
                local health=( $(job_health "$job_id") )
                local hstat=${health[0]}
                local hmsg=${health[@]:1}
                local gpu_hours=$(awk -v g=$gpu_count -v e=$elapsed 'BEGIN{printf "%.2f", (g*e)/3600.0}')
                local act=$(get_current_activity "$job_id")
                local tput=$(compute_throughput "$elapsed" "$completed")
                local eta_str="--"; if (( eta > 0 )); then eta_str=$(printf "%02dh%02dm" $((eta/3600)) $(((eta%3600)/60))); fi
                echo "Job $job_id ($name) - $state for $runtime on $node | GPU=${gpu_type}x${gpu_count} used~${gpu_hours} GPUh | Progress=${completed}/${total} (~${tput} cfg/h) | ETA~${eta_str} finish=$finish_str | Health=$hstat ${hmsg} | ${act}"
                if [[ $total -gt 0 ]]; then
                    echo -n "  Progress: "
                    progress_bar $completed $total
                    echo " ($successful successful)"
                else
                    echo "  Status: Starting..."
                fi
            else
                echo "Job $job_id: Completed or not found"
            fi
            echo ""
        done
        
        echo "Press Ctrl+C to exit | Refresh: ${refresh}s"
        sleep $refresh
    done
}

# Main function
main() {
    local mode="live"
    local job_ids=()
    local refresh=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --live|-l) mode="live"; shift ;;
            --quick|-q) mode="quick"; shift ;;
            --multi|-m) mode="multi"; shift ;;
            --refresh) refresh="$2"; REFRESH_INTERVAL="$2"; shift 2 ;;
            --help|-h) show_help; exit 0 ;;
            *)
                if [[ "$1" =~ ^[0-9]+$ ]]; then
                    job_ids+=("$1")
                else
                    error "Unknown option: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    case $mode in
        live) show_live ;;
        quick) show_quick ;;
        multi)
            if [[ ${#job_ids[@]} -eq 0 ]]; then
                # Auto-detect ICL jobs
                mapfile -t job_ids < <(get_icl_jobs | awk '{print $1}')
                if [[ ${#job_ids[@]} -eq 0 ]]; then
                    error "No ICL jobs found"
                    exit 1
                fi
            fi
            show_multi "${job_ids[@]}"
            ;;
    esac
}

main "$@"