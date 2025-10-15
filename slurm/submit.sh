#!/bin/bash
# ICL Time Series - SLURM Job Submission
# Clean, minimal job submission for high-throughput experiments

set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Utility functions
log() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}" >&2; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

show_help() {
    cat << EOF
ICL Time Series - SLURM Job Submission

USAGE:
    $0 [OPTIONS] [--multi GPU_TYPE COUNT]

OPTIONS:
    --workers N        Parallel workers (default: $DEFAULT_WORKERS)
    --memory SIZE      Memory allocation (default: $DEFAULT_MEMORY)
    --time DURATION    Time limit (default: $DEFAULT_TIME)
    --limit N          Limit total configs to N (slices evenly across jobs)
    --dry-run          Show what would be submitted
    --multi GPU COUNT  Multi-GPU submission
    --resume           Run only missing models
    --help             Show this help

EXAMPLES:
    $0 --multi a5000 4
    $0 --resume --multi v100 2
    $0 --limit 4 --multi a6000 1
EOF
}

# Validate environment
validate() {
    command -v sbatch >/dev/null || { error "SLURM not available"; exit 1; }
    [[ -d "$EXPERIMENTS_DIR" && -d "$CONFIGS_DIR" ]] || { error "Invalid project structure"; exit 1; }
    
    # Check for existing jobs
    local existing=$(squeue -u "$USER" -o "%.18i %.8j" --noheader 2>/dev/null | grep "icl_" | head -1 || echo "")
    if [[ -n "$existing" ]]; then
        warn "Existing ICL job found: $existing"
    fi
}

# Create SLURM script
create_script() {
    local gpu_type="$1"
    local workers="$2"
    local start_idx="$3"
    local end_idx="$4"
    local resume_file="$5"
    
    # Resolve profile locally because associative arrays cannot be exported
    local profile="8:64G:24:00:00"
    if [[ -n "${GPU_PROFILES[$gpu_type]+set}" ]]; then
        profile="${GPU_PROFILES[$gpu_type]}"
    fi
    IFS=':' read -r cpus mem time <<< "$profile"
    
    cat << EOF
#!/bin/bash
#SBATCH --job-name=icl_${gpu_type}_${start_idx}_${end_idx}
#SBATCH --gres=gpu:${gpu_type}:1
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH --export=ALL
#SBATCH --output=$LOGS_DIR/icl_${gpu_type}_%j.out
#SBATCH --error=$LOGS_DIR/icl_${gpu_type}_%j.err

source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torchpy310

# Snapshot GPU memory at start (first GPU)
if command -v nvidia-smi >/dev/null 2>&1; then
  ts=\$(date '+%Y-%m-%d %H:%M:%S')
  mem_line=\$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -n1 || true)
  if [[ -n "\$mem_line" ]]; then
    gpu_name=\$(echo "\$mem_line" | awk -F, '{gsub(/^ +| +$/,"",$1); print $1}')
    mem_total=\$(echo "\$mem_line" | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')
    mem_used=\$(echo "\$mem_line" | awk -F, '{gsub(/^ +| +$/,"",$3); print $3}')
    echo "[\${ts}] GPU: \${gpu_name} | GPU_MEM_START=\${mem_used}/\${mem_total} MiB"
  fi
fi

# Propagate optional quick-test epochs if provided in submit environment
if [[ -n "\${ICL_TRAIN_EPOCHS:-}" ]]; then
  export ICL_TRAIN_EPOCHS="\${ICL_TRAIN_EPOCHS}"
fi

cd "$PROJECT_ROOT"
mkdir -p "$LOGS_DIR" "$EXPERIMENTS_DIR"/{context_scaling,lsa_layers}/{checkpoints,results,plots}

# Prime monitor with total config count for ETA before Python starts
echo "Running $(( end_idx - start_idx )) configurations"

python -u "$SCRIPT_DIR/scripts/run_experiments.py" \
    --experiments-dir "$EXPERIMENTS_DIR" \
    --configs-dir "$CONFIGS_DIR" \
    --start-idx $start_idx \
    --end-idx $end_idx \
    --parallel-workers $workers \
    --device cuda \
    ${resume_file:+--missing-configs-file "$resume_file"}
EOF
}

# Submit multi-GPU jobs
submit_multi() {
    local gpu_type="$1"
    local gpu_count="$2"
    local workers="$3"
    local dry_run="$4"
    local resume="$5"
    local limit_total="$6"
    
    log "Submitting $gpu_count jobs on $gpu_type GPUs"
    
    # Handle resume mode
    local resume_file=""
    local total_models=$TOTAL_MODELS
    if [[ "$resume" == "true" ]]; then
        log "Resume mode: finding missing models..."
        mkdir -p "$LOGS_DIR"
        resume_file="$LOGS_DIR/missing_configs.json"
        python "$SCRIPT_DIR/scripts/check_missing_models.py" \
            --configs-dir "$CONFIGS_DIR" \
            --experiments-dir "$EXPERIMENTS_DIR" \
            --output-file "$resume_file" \
            --min-epochs 150 1>/dev/null || true
        
        if [[ ! -s "$resume_file" ]]; then
            success "No missing models found"
            rm -f "$resume_file"
            return 0
        fi
        total_models=$(python -c "import json,sys; print(len(json.load(open(\"$resume_file\"))))" 2>/dev/null || echo "0")
        log "Found $total_models missing models"
        if [[ "$total_models" == "0" ]]; then
            success "All experiments completed"
            rm -f "$resume_file"
            return 0
        fi
    fi
    
    # Apply limit if provided
    if [[ -n "$limit_total" && "$limit_total" != "0" ]]; then
        total_models="$limit_total"
    fi
    
    local configs_per_job=$(( (total_models + gpu_count - 1) / gpu_count ))
    local job_ids=()
    
    for i in $(seq 1 "$gpu_count"); do
        local start_idx=$(( (i-1) * configs_per_job ))
        local end_idx=$((start_idx + configs_per_job))
        [[ $end_idx -gt $total_models ]] && end_idx=$total_models
        
        local actual_workers=$workers
        [[ $((end_idx - start_idx)) -lt $workers ]] && actual_workers=$((end_idx - start_idx))
        
        if [[ $((end_idx - start_idx)) -eq 0 ]]; then
            continue
        fi
        
        log "Job $i: models $start_idx-$end_idx with $actual_workers workers"
        
        if [[ "$dry_run" == "true" ]]; then
            echo "DRY RUN: Would submit job for models $start_idx-$end_idx"
            continue
        fi
        
        # Create and submit script
        local script_file=$(mktemp)
        create_script "$gpu_type" "$actual_workers" "$start_idx" "$end_idx" "$resume_file" > "$script_file"
        
        local job_id=$(sbatch "$script_file" 2>&1 | grep -o '[0-9]\+' | tail -1)
        job_ids+=("$job_id")
        success "Submitted job $job_id"
        
        rm "$script_file"
    done
    
    # Keep resume_file for running jobs; do not delete
    
    if [[ ${#job_ids[@]} -gt 0 ]]; then
        success "All jobs submitted: ${job_ids[*]}"
        log "Monitor with: slurm/monitor.sh --multi ${job_ids[*]}"
    fi
}

# Main function
main() {
    local workers=$DEFAULT_WORKERS
    local memory=$DEFAULT_MEMORY
    local time_limit=$DEFAULT_TIME
    local dry_run="false"
    local resume="false"
    local limit_total=""
    local gpu_type=""
    local gpu_count=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --workers) workers="$2"; shift 2 ;;
            --memory) memory="$2"; shift 2 ;;
            --time) time_limit="$2"; shift 2 ;;
            --limit) limit_total="$2"; shift 2 ;;
            --multi) gpu_type="$2"; gpu_count="$3"; shift 3 ;;
            --resume) resume="true"; shift ;;
            --dry-run) dry_run="true"; shift ;;
            --help|-h) show_help; exit 0 ;;
            *) error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done
    
    validate
    
    if [[ -n "$gpu_type" && -n "$gpu_count" ]]; then
        submit_multi "$gpu_type" "$gpu_count" "$workers" "$dry_run" "$resume" "$limit_total"
    else
        error "Multi-GPU mode requires --multi GPU_TYPE COUNT"
        exit 1
    fi
}

main "$@"