# 🚀 SLURM Job Management & Distributed Computing

SLURM job orchestration for ICL Time Series experiments. Submit slices of the configuration space across multiple GPUs, monitor progress live, and optionally resume only missing models.

---

## 📁 Directory Structure

```
slurm/
├── README.md                    # This file
├── config.sh                    # Central configuration and totals
├── submit.sh                    # Job submission across GPUs
├── monitor.sh                   # Live/quick/multi-job monitoring
└── scripts/
    ├── run_experiments.py       # Entrypoint used by submit.sh
    ├── generate_configs.py      # Generates configs from YAML
    └── check_missing_models.py  # Finds missing/incomplete models
```

---

## 🎯 Core Purpose

- Distribute experiments across multiple GPUs with one command
- Track live progress, ETA, throughput, and basic health
- Resume only missing or under-trained configurations

---

## 🚀 Quick Start

```bash
# 1) Submit experiments (2 A5000 GPUs, 8 workers per job)
slurm/submit.sh --multi a5000 2 --workers 8

# 2) Monitor progress
slurm/monitor.sh           # live dashboard
slurm/monitor.sh --quick   # snapshot

# 3) Count results
find experiments -name "results.json" | wc -l
```

Notes:
- The submission script schedules one GPU per job. Training runs sequentially per job (GPU-bound), while the Python runner coordinates the configuration slice.
- You can limit the total number of configs for smoke tests with `--limit N`.
- For quick tests, set `ICL_TRAIN_EPOCHS=5` before submission to reduce epochs.

---

## 📋 Script Overview

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `submit.sh` | Job submission with slicing | `--multi GPU_TYPE COUNT` (required), `--workers N`, `--memory SIZE`, `--time D-HH:MM:SS`, `--limit N`, `--resume`, `--dry-run` |
| `monitor.sh` | Live/quick/multi-job status | `--quick`, `--multi JOB_IDS...`, `--refresh N` |
| `config.sh` | Central config and totals | Defines paths, totals, default workers/mem/time, GPU profiles |
| `scripts/run_experiments.py` | Runs a config slice | `--experiments-dir`, `--configs-dir`, `--start-idx`, `--end-idx`, `--parallel-workers`, `--device` |
| `scripts/check_missing_models.py` | Missing models for resume | `--min-epochs` to require a minimum training length |

---

## 🎮 Usage Examples

### Submit jobs
```bash
# Distribute across 4 V100 GPUs
slurm/submit.sh --multi v100 4 --workers 8

# Limit total configs to 6 (smoke test)
slurm/submit.sh --multi a6000 1 --limit 6 --workers 4

# Resume mode: run only missing/under-trained models (min epochs=150)
slurm/submit.sh --resume --multi a5000 2
```

### Monitor jobs
```bash
# Live dashboard for all ICL jobs
slurm/monitor.sh

# Quick status
slurm/monitor.sh --quick

# Multi-job monitor for specific job IDs
slurm/monitor.sh --multi 12345 12346

# Refresh interval (seconds)
slurm/monitor.sh --refresh 10
```

ETA semantics: The monitor estimates remaining time from per-config throughput and completed count (not walltime). It approximates the time to finish the current slice on the node.

---

## 🏗️ How it Works

1) `config.sh` computes totals from `configs/` (fallback defaults are used if Python is unavailable).
2) `submit.sh` slices the config space across `COUNT` jobs for the chosen `GPU_TYPE` and submits one GPU per job using the profile in `config.sh`.
3) Each job runs `scripts/run_experiments.py` to execute its slice on `cuda`.
4) `monitor.sh` queries SLURM and logs to show live progress, ETA, and basic health.

Training is GPU-bound and runs sequentially per job to avoid CUDA threading issues. The runner caches datasets and writes per-model `results.json` next to `best_model.pt`.

---

## ⚙️ Configuration (config.sh)

Exported variables:
- `PROJECT_ROOT`, `EXPERIMENTS_DIR`, `LOGS_DIR`, `CONFIGS_DIR`
- `CONTEXT_TOTAL`, `LSA_TOTAL`, `TOTAL_MODELS` (auto-computed)
- `DEFAULT_WORKERS` (8), `DEFAULT_MEMORY` (32G), `DEFAULT_TIME` (24:00:00)

GPU profiles (editable in `config.sh`):
```bash
# Format: cpus:mem:time
rtx_2080: 8:64G:24:00:00
a5000:    8:64G:24:00:00
a6000:    8:128G:24:00:00
v100:     8:64G:24:00:00
gpu:      8:64G:24:00:00
```

To add a GPU type, extend the `GPU_PROFILES` map in `config.sh`:
```bash
declare -Ag GPU_PROFILES=(
  [new_gpu]="8:64G:24:00:00"
)
```

---

## 🔁 Resume Mode

`submit.sh --resume` calls `scripts/check_missing_models.py` to compute a list of missing/under-trained configs (by default `min-epochs=150`). The submission then runs only those configs, split evenly across jobs.

---

## 🧪 Environment & Logs

The submitted job script activates a Conda environment named `torchpy310`. Adjust it in `submit.sh` if your environment differs.

Logs are written to `logs/` as `icl_<gpu>_<jobid>.out` and `icl_<gpu>_<jobid>.err`. The monitor surfaces recent activity and basic health (traceback detection, stale logs).

---

## 🚨 Troubleshooting

```bash
# Verify SLURM is available
which sbatch && which squeue && which scontrol

# No jobs showing
squeue -u $(whoami)

# Inspect logs
ls logs/
tail -n 200 logs/icl_*_*.out

# Debug resume list
python slurm/scripts/check_missing_models.py --help
```

---

## 📈 Tips

- Use `--limit N` and `ICL_TRAIN_EPOCHS=5` for quick smoke tests.
- Prefer consistent seeds across GPUs for reproducibility.
- If your cluster lacks a specific GPU profile key, add it in `config.sh` or use the generic `gpu` profile.
