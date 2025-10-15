#!/usr/bin/env python3
"""
Single-run/SLURM helper for ICL Time Series.

Modes:
  1) Default single-run training (train_lsa.py)
  2) Softmax comparison experiment (compare_softmax.py) via --compare-softmax

Output structure for compare-softmax (in current working directory):
  compare_softmax/seed{seed}_p{p}_n{n}_L{L}/
    ├── shared/train_series.npy, test_series.npy (single copy)
    ├── linear/best_model.pt, model_config.json
    └── softmax/best_model.pt, model_config.json

Examples (local GPU):
  python experiments/single_run_slurm.py --p 7 --layers 2 --history-len 15
  python experiments/single_run_slurm.py --compare-softmax --p 7 --layers 2 --history-len 15 --seed 42

Submit to SLURM (GPU):
  python experiments/single_run_slurm.py --p 7 --layers 2 --history-len 15 --submit
  python experiments/single_run_slurm.py --compare-softmax --p 7 --layers 2 --history-len 15 --seed 42 --submit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure repo root is on sys.path before importing project modules (works from experiments/)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run trainer (no eval)")
    parser.add_argument("--compare-softmax", action="store_true", help="Run softmax vs linear comparison experiment")
    # Model/data
    parser.add_argument("--p", type=int, required=True, help="AR order / context length")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSA layers")
    parser.add_argument("--history-len", type=int, default=None, help="History length (defaults to p+2 if None)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise std for AR generator")
    parser.add_argument("--series-len", type=int, default=50000, help="Total length of the synthetic series")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train split fraction")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--use-softmax", action="store_true", help="Enable softmax attention (default: linear LSA)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for both modes)")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (compare-softmax mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda|cpu)")

    # Output
    parser.add_argument("--out", type=str, default="single_run",
                        help="Output artifacts directory (default: single_run)")
    parser.add_argument("--subdir", type=str, default="compare_softmax", help="Subdir name for experiments (created in current directory)")

    # SLURM options
    parser.add_argument("--submit", action="store_true", help="Submit as a SLURM GPU job")
    parser.add_argument("--gpu-type", type=str, default="a5000", help="GPU type for SLURM (default: a5000)")
    parser.add_argument("--time", type=str, default="12:00:00", help="Time limit for SLURM job")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task for SLURM job")
    parser.add_argument("--mem", type=str, default="32G", help="Memory for SLURM job")
    return parser.parse_args()


def build_cmd(args: argparse.Namespace) -> list[str]:
    if args.compare_softmax:
        script = repo_root / "experiments" / "compare_softmax.py"
        cmd = [
            sys.executable,
            str(script),
            "--p", str(args.p),
            "--layers", str(args.layers),
            "--sigma", str(args.sigma),
            "--series-len", str(args.series_len),
            "--train-split", str(args.train_split),
            "--val-split", str(args.val_split),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--device", str(args.device),
            "--subdir", str(args.subdir),
        ]
        if args.history_len is not None:
            cmd.extend(["--history-len", str(args.history_len)])
        cmd.extend(["--seed", str(args.seed)])
        return cmd
    else:
        train_script = repo_root / "experiments" / "train_lsa.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--p", str(args.p),
            "--layers", str(args.layers),
            "--use-softmax" if args.use_softmax else "",
            "--sigma", str(args.sigma),
            "--series-len", str(args.series_len),
            "--train-split", str(args.train_split),
            "--val-split", str(args.val_split),
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", str(args.device),
            "--out", str(args.out),
        ]
        # Remove any empty strings from optional flags
        cmd = [c for c in cmd if c != ""]
        if args.history_len is not None:
            cmd.extend(["--history-len", str(args.history_len)])
        return cmd


def run_local(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def submit_slurm(cmd: list[str], args: argparse.Namespace) -> int:
    # Put logs in parent directory (project root), not under experiments/
    logs_dir = Path("../logs")
    logs_dir.mkdir(exist_ok=True)
    # Ensure we use the conda environment's python (not an absolute path captured at submission)
    cmd_for_wrap = cmd.copy()
    cmd_for_wrap[0] = "python"
    wrapped = " ".join(cmd_for_wrap).replace("\"", "\\\"")
    job_name = (
        f"icl_compare_softmax_p{args.p}_L{args.layers}"
        if args.compare_softmax else f"icl_single_run_p{args.p}_L{args.layers}"
    )
    sbatch_cmd = [
        "sbatch",
        "--job-name", job_name,
        "--gres", f"gpu:{args.gpu_type}:1",
        "--cpus-per-task", str(args.cpus),
        "--mem", str(args.mem),
        "--time", str(args.time),
        "--output", str(logs_dir / "icl_single_run_%j.out"),
        "--error", str(logs_dir / "icl_single_run_%j.err"),
        "--wrap",
        (
            "bash -lc \""
            "source $HOME/miniconda3/etc/profile.d/conda.sh && "
            "conda activate torchpy310 && "
            f"{wrapped}"
            "\""
        ),
    ]
    print("Submitting:", " ".join(sbatch_cmd))
    proc = subprocess.run(sbatch_cmd, text=True, capture_output=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip())
    return proc.returncode


def main() -> int:
    args = parse_args()
    cmd = build_cmd(args)
    if args.submit:
        return submit_slurm(cmd, args)
    return run_local(cmd)


if __name__ == "__main__":
    raise SystemExit(main())


