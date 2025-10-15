#!/bin/bash
# ICL Time Series - SLURM Configuration
# Clean, minimal configuration for high-throughput experiments

# Core paths
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
readonly LOGS_DIR="$PROJECT_ROOT/logs"
readonly CONFIGS_DIR="$PROJECT_ROOT/configs"

# Experiment totals (calculated dynamically)
calculate_totals() {
    python -c "
import sys
from pathlib import Path
sys.path.append('$PROJECT_ROOT')
from utils.config import generate_all_configs
configs = generate_all_configs('$CONFIGS_DIR')
context = [c for c in configs if c['experiment'] == 'context_scaling']
lsa = [c for c in configs if c['experiment'] == 'lsa_layers']
print(f'{len(context)} {len(lsa)} {len(configs)}')
" 2>/dev/null || echo "30 18 48"
}

readonly TOTALS=$(calculate_totals)
readonly CONTEXT_TOTAL=$(echo "$TOTALS" | awk '{print $1}')
readonly LSA_TOTAL=$(echo "$TOTALS" | awk '{print $2}')
readonly TOTAL_MODELS=$(echo "$TOTALS" | awk '{print $3}')

# Default settings
readonly DEFAULT_WORKERS=8
readonly DEFAULT_MEMORY="32G"
readonly DEFAULT_TIME="24:00:00"

# GPU profiles
declare -Ag GPU_PROFILES=(
    [rtx_2080]="8:64G:24:00:00"
    [a5000]="8:64G:24:00:00"
    [a6000]="8:128G:24:00:00"
    [v100]="8:64G:24:00:00"
    [gpu]="8:64G:24:00:00"
)

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Export variables
export PROJECT_ROOT EXPERIMENTS_DIR LOGS_DIR CONFIGS_DIR
export CONTEXT_TOTAL LSA_TOTAL TOTAL_MODELS
export DEFAULT_WORKERS DEFAULT_MEMORY DEFAULT_TIME
# Do NOT export associative arrays (unsupported); keep GPU_PROFILES in-shell only