# Why Do Transformers Fail to Forecast Time Series In-Context?

[![arXiv](https://img.shields.io/badge/arXiv-2510.09776-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.09776)
[![GitHub](https://img.shields.io/badge/GitHub-ICL--Time--Series-181717?logo=github)](https://github.com/MasterZhou1/ICL-Time-Series)

**Authors:**
[**Yufa Zhou***](https://masterzhou1.github.io/)$^1$, [**Yixiao Wang***](https://yixiao-wang-stats.github.io/)$^1$,
[**Surbhi Goel**](https://www.surbhigoel.com/)$^2$, [**Anru R. Zhang**](https://anruzhang.github.io/)$^1$

$^1$**Duke University**, $^2$**University of Pennsylvania**

*Equal contribution

## Introduction

Despite their widespread success in language and vision, Transformers often underperform simple linear models on time series forecasting (TSF) tasks. While empirical studies have consistently shown this, the underlying theoretical reason remained unclear.
This work provides a rigorous formal analysis explaining **why** Transformers exhibit those limitations through the lens of In-Context Learning (ICL) theory.

### Theory

We analyze Linear Self-Attention (LSA) under AR(p) data and establish:

- LSA cannot outperform classical linear predictors in expected mean-squared error (MSE).

- A strict finite-sample performance gap exists between LSA and the optimal linear forecaster, vanishing no faster than 1/n as the context length grows.

- Under Chain-of-Thought inference, Transformer rollouts collapse exponentially to the mean, compounding prediction errors over time.

These results show that attention mechanisms have fundamental representational limitations for TSF, which explains why they fail to outperform OLS even under ideal training conditions. See full details in [Paper](https://arxiv.org/abs/2510.09776).


### Codebase Overview
This repository accompanies the paper “[Why Do Transformers Fail to Forecast Time Series In-Context?](https://arxiv.org/pdf/2510.09776)” and provides code to simulate stationary AR(p) time series, train Linear Self-Attention (LSA) Transformers with Hankel embeddings, compare against a classical OLS AR baseline, evaluate both teacher-forcing and chain-of-thought predictions, and generate pretty plots. See the paper for detailed theory and proofs. 



### Key components
- **Data**: `data/ar_sims.py` generates weakly stationary AR(p) series (roots outside unit circle) and provides diagnostics; `data/ar_dataloader.py` builds (history, target) pairs with `history_len > p`.
- **Models**: `models/lsa_transformer.py` implements LSA layers and an `LSATransformerWithHankel` wrapper (Hankel construction + prediction APIs); `models/linear_ar.py` is an OLS AR(p) baseline.
- **Training/Eval**: `experiments/train_lsa.py`, `experiments/compare_softmax.py`, `experiments/evaluate_models.py`.
- **Plotting**: `experiments/generate_plots.py`, `utils/plotting.py`, configured by `configs/plots.yaml`.
- **Batch runs (SLURM)**: `slurm/submit.sh`, `slurm/monitor.sh`, with experiment orchestration via `utils/experiment_runner.py`.

---

## Installation
1) Clone and install dependencies:
```bash
pip install -r requirements.txt
```
2) (Optional) Weights & Biases logging. Create a `.env` with:
```bash
WANDB_MODE=online
WANDB_API_KEY=...your_key...
WANDB_PROJECT=icl-time-series
```
If unset, logging is disabled by default (`utils/wandb_utils.py`).

## Quick start (local)
Train an LSA Transformer on a synthetic AR series and save artifacts to `artifacts/`:
```bash
python experiments/train_lsa.py --p 7 --layers 1 --history-len 9 --epochs 100
```
CLI (selected flags; see the script for full defaults):
- `--p` (int): AR order / context length. Default: 7
- `--history-len` (int): window for histories; must satisfy `history-len > p`. Default: `p + 2`
- `--layers` (int): number of stacked LSA layers. Default: 1
- `--sigma` (float): noise std for AR generator. Default: 0.1
- `--series-len` (int): total series length. Default: 50000
- `--train-split`/`--val-split`: fractions; test is auto-computed. Defaults: 0.7/0.15
- `--epochs`/`--batch-size`/`--lr`/`--patience`: training hyperparameters
- `--use-softmax`: switch from linear LSA to softmax attention
- `--out` (str): output dir. Default: `artifacts`
- `--data-dir` (str): load shared `train_series.npy` and `test_series.npy` instead of regenerating

Artifacts written to the output directory:
- `best_model.pt`, `model_config.json`, `train_series.npy`, `test_series.npy`, and an enriched `results.json` with metrics and run metadata.

Evaluate on saved artifacts (and compare to OLS AR):
```bash
python experiments/evaluate_models.py --artifacts-dir artifacts --cot-steps 50
```
Outputs: printed MSEs and PDFs in the artifacts directory (e.g., `tf_values_*.pdf`, `cot_values_*.pdf`).

Compare Linear LSA vs Softmax under identical settings:
```bash
python experiments/compare_softmax.py --p 5 --layers 1 --history-len 8 --seed 42
python experiments/evaluate_models.py \
  --compare-softmax experiments/compare_softmax/seed42_p5_n8_L1 --cot-steps 50
```
The `compare_softmax.py` script writes one dataset under `shared/` and two checkpoints under `lsa/` and `softmax/` within the run directory.

## Experiments and configuration
Experiments are driven by YAML in `configs/` with inheritance via `utils/config.load_config`:
- `configs/base_config.yaml`: common data, training, evaluation, and output directories.
- `configs/context_scaling.yaml`: vary `p` and `history_len` (via offsets), fixed LSA layers.
- `configs/lsa_layers.yaml`: vary number of LSA layers at fixed `history_len`.

To aggregate results and generate plots from completed runs:
```bash
python experiments/generate_plots.py --experiment-type all
```
Customize bands/outliers/exclusions via `configs/plots.yaml`.

## SLURM (recommended for GPUs)
Submit a multi-GPU sweep (slices config space evenly across jobs):
```bash
slurm/submit.sh --multi a5000 4 --workers 8
```
Options:
- `--workers N`: per-job parallel workers used by the Python runner (training itself is GPU-bound and runs sequentially per job for stability).
- `--limit N`: cap total configs (useful for smoke tests).
- `--resume`: run only missing/incomplete configs discovered by `slurm/scripts/check_missing_models.py`.
- `ICL_TRAIN_EPOCHS=5 ...`: environment override to reduce epochs for quick tests.

Monitor jobs:
```bash
slurm/monitor.sh          # live dashboard
slurm/monitor.sh --quick  # snapshot
```
Note: the ETA shown by the monitor is computed from training progress and completed configs (not walltime); it is an estimate of remaining time to finish the queued slice on the node.

## Data and baselines
- `data/ar_sims.ARDataGenerator(p, sigma, sequence_length, random_seed)`: generates weakly stationary AR(p) series in NumPy. Stationarity is enforced by sampling coefficients whose characteristic roots lie outside the unit circle; stationary variance is computed via Yule–Walker. Methods: `generate_series`, `generate_multiple_series`, `prepare_data`, `get_stationarity_info`.
- `data/ar_dataloader.ARForecastDataset(series, p, history_len)`: returns `(history, target)` tensors for one-step forecasting with the constraint `history_len > p` (the Hankel path needs at least two columns).
- `models/linear_ar.LinearARModel(p)`: OLS baseline with `fit`, `predict`, and `predict_chain_of_thought` (autoregressive) utilities.

## LSA Transformer (Hankel)
`models/lsa_transformer.py` implements:
- `LinearSelfAttention`: linear attention update with optional softmax; expects input `H ∈ R^{B×(d+1)×(n+1)}` and applies a causal mask on the `(n+1)` dimension.
- `LSATransformer`: stacks LSA layers and exposes `predict` (reads the final token’s last coordinate).
- `LSATransformerWithHankel`: wraps time series to Hankel matrices of shape `(B×(p+1)×(n_hankel))` and exposes convenient APIs:
  - `predict(sequences)`
  - `predict_teacher_forcing(sequences)`
  - `predict_chain_of_thought(initial_sequences, steps)`

## Evaluation utilities
`experiments/evaluate_models.py` loads artifacts and computes MSE for both teacher-forcing and chain-of-thought. Key flags:
- `--artifacts-dir DIR`: single-model evaluation (expects `best_model.pt`, `model_config.json`, and `train/test_series.npy`).
- `--compare-softmax DIR`: evaluate a `compare_softmax.py` run directory containing `shared/`, `lsa/`, and `softmax/`.
- `--cot-steps K`: number of autoregressive steps (capped by test length).
- `--no-plots`: skip PDF generation.

## Reproducibility and logging
- Seeding: `utils/common.set_global_seed` sets `random`, `numpy`, and `torch` (deterministic modes when available).
- W&B: `utils/wandb_utils.WandbRun` automatically reads `.env`; runs default to disabled unless `WANDB_MODE=online` and a valid key are provided.

## Artifacts and layout
By default, experiment outputs are organized under `experiments/` according to `configs/*`:
```
experiments/
├── context_scaling/
│   ├── checkpoints/   # one subdir per (seed, p, history_len, layers)
│   ├── results/       # CSVs aggregated by utils/plotting.py
│   └── plots/         # PDFs
└── lsa_layers/
    ├── checkpoints/
    ├── results/
    └── plots/
```
Individual model dirs include: `best_model.pt`, `model_config.json`, `results.json`, and cached `train/test_series.npy` when applicable.

## Project structure
```
configs/        # YAML configs (base, context_scaling, lsa_layers, plots)
data/           # AR generator + forecasting dataset
experiments/    # Training, evaluation, plotting, softmax comparison
models/         # LSA Transformer + OLS AR baseline
slurm/          # submit.sh, monitor.sh, helper scripts
utils/          # config loader, experiment runner, plotting, wandb utils, common
```

## Citation
If you find this repository useful, please cite:
```
@article{zhou2025tsf,
  title   = {Why Do Transformers Fail to Forecast Time Series In-Context?},
  author  = {Zhou, Yufa and Wang, Yixiao and Goel, Surbhi and Zhang, Anru R.},
  journal = {arXiv preprint arXiv:2510.09776},
  year    = {2025}
}
```

## Contact

😊 Questions? Ideas? Interested in collaborating on this exciting research?  

Feel free to reach out to Yufa Zhou at [yufa.zhou@duke.edu](mailto:yufa.zhou@duke.edu)—always happy to connect!



