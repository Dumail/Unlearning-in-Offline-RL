# TOUR — A Benchmark of Trajectory-level memOrization and Unlearning in offline RL

This release contains the implementation of our TOUR benchmark to reproduce the **results** of paper.

---

## 1. Installation

The repository is managed by [`uv`](https://docs.astral.sh/uv/). Python 3.12+ is required.

```bash
uv sync                         # installs runtime + dev (pytest) groups
```

If your environment does not have CUDA 12.4, replace the `[tool.uv.sources] torch = { index = "pytorch-cu124" }` line in `pyproject.toml` with the index that matches your platform (e.g. `pytorch-cpu` for `https://download.pytorch.org/whl/cpu`) before running `uv sync`.

The benchmark uses `gymnasium[mujoco]` to evaluate D4RL normalized scores. On headless servers, ensure `MUJOCO_GL=egl` (or another headless backend) is set in your environment if rendering issues arise.

---

## 2. Data preparation

Download the D4RL v2 HDF5 files for the 9 locomotion settings used in the main benchmark and place them under `data/gym_mujoco_v2/`:

```text
data/gym_mujoco_v2/
├── halfcheetah-medium-replay-v2/halfcheetah_medium_replay-v2.hdf5
├── halfcheetah-medium-v2/halfcheetah_medium-v2.hdf5
├── halfcheetah-medium-expert-v2/halfcheetah_medium_expert-v2.hdf5
├── hopper-medium-replay-v2/hopper_medium_replay-v2.hdf5
├── hopper-medium-v2/hopper_medium-v2.hdf5
├── hopper-medium-expert-v2/hopper_medium_expert-v2.hdf5
├── walker2d-medium-replay-v2/walker2d_medium_replay-v2.hdf5
├── walker2d-medium-v2/walker2d_medium-v2.hdf5
└── walker2d-medium-expert-v2/walker2d_medium_expert-v2.hdf5
```

Then run for each environment (or use `run_main_reproduction.py --prep-data`):

```bash
uv run python scripts/download_data.py env=hopper_mr
uv run python scripts/build_negative_set.py env=hopper_mr
```

The download script will read the HDF5 file under `data/gym_mujoco_v2/`, partition trajectories (70% train / 15% calibration / 15% test, 10% of train as forget set), and write the splits + matched negative set to `data/<env_full>/`.

---

## 3. Reproducing the results

The single-entry orchestrator is `scripts/run_main_reproduction.py`. It runs 6 stages in order; previewing the task list with `--dry-run` first is recommended:

```bash
# Preview
uv run python scripts/run_main_reproduction.py --dry-run

# Run end-to-end (data must already be prepared)
uv run python scripts/run_main_reproduction.py

# Or run a subset of stages
uv run python scripts/run_main_reproduction.py --stages dt_baselines analyze figures
```

Stages:

| Stage | Workload | Approx wall time on a single GTX 4070 Ti Super |
|---|---|---|
| `dt_baselines` | 9 settings × 3 seeds × {B1, B2, B3, B4} + TMI re-export | ~8 h |
| `trajdeleter` | 9 settings × 3 seeds × B4T canonical | ~2 h |
| `backbone` | 3 replay envs × 3 seeds × {DT, MLP, LSTM} matched-budget | ~3 h |
| `selective` | 3 replay envs × 3 seeds × 5 targets × 3 ascent steps | ~6 h |
| `analyze` | 7 analysis scripts that aggregate JSON → CSV/JSON summaries | < 5 min |
| `figures` | `figures/gen_tables_main.py` + `figures/gen_fig2_heatmap.py` | < 30 s |

Outputs land under `results/`:

- Per-run JSON: `results/<env_full>/{tmi_eval_dt_final, gold_standard_seed{0,1,2}, naive_ft_seed{0,1,2}, ga_refit_lambda0.1_steps500_seed{0,1,2}, trajdeleter_alpha1.0_beta2.0_s1100_s21000_seed{0,1,2}, b6_{dt,mlp,lstm}_summary}.json`
- Selective per-run JSON: `results/selective/<env_full>/selective_<target>_lambda1.0_steps<steps>_seed<seed>.json`
- Aggregated analysis: `results/analysis/{benchmark_audit, b1_b4_multiseed_summary, multi_attack_high_utility/multi_attack_summary, b6_matched_budget_multiseed, selective_summary, selective_utility_budget/selective_utility_budget_summary, layer_selection_cv}.{json,csv,md}`
- Generated paper assets: `figures/TABLE_*.txt`, `figures/fig2_privacy_gap_heatmap.{pdf,png}`

To run a single experiment by hand (Hydra CLI overrides):

```bash
uv run python scripts/train_base_dt.py env=hopper_mr seed=0
uv run python scripts/run_unlearning.py env=hopper_mr seed=0 unlearn.kl_weight=0.1 unlearn.ascent_steps=500
uv run python scripts/run_trajdeleter_unlearning.py env=hopper_mr seed=0 unlearn=trajdeleter unlearn.beta=2.0 unlearn.stage1_steps=100 unlearn.stage2_steps=1000
uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=attn.layer_1 unlearn.ascent_steps=250
```

---

## 4. Directory layout

```text
review_release/
├── README.md                      this file
├── pyproject.toml                 dependencies (uv-managed)
├── uv.lock                        pinned resolution
├── .python-version                3.12
├── configs/                       Hydra configs (env / model / train / unlearn)
├── src/                           library code (DT, MLP, LSTM, TMI, matching, unlearning)
├── scripts/                       Hydra/argparse CLI entry points
│   ├── run_main_reproduction.py   ← top-level orchestrator
│   ├── run_release_smoke.py       ← end-to-end pipeline smoke
│   ├── train_base_dt.py · run_gold_standard.py · run_naive_ft.py
│   ├── run_unlearning.py · run_trajdeleter_unlearning.py
│   ├── run_backbone_comparison.py · run_selective_unlearning.py
│   ├── evaluate_tmi.py · build_negative_set.py · download_data.py
│   ├── run_layer_selection_cv.py
│   ├── build_benchmark_audit.py · analyze_b1_b4_multiseed.py
│   ├── analyze_multi_attack.py · analyze_b6_matched_budget.py
│   ├── analyze_selective_results.py · analyze_selective_utility_budget.py
├── figures/                       paper-asset generators + concept image
│   ├── gen_tables_main.py         ← produces 4 main-text plain-text tables
│   ├── gen_fig2_heatmap.py        ← produces fig:heatmap
│   └── table_utils.py · paper_plot_style.py
├── data/.gitkeep                  user supplies HDF5 files here
├── checkpoints/.gitkeep           per-run training artifacts
└── results/.gitkeep               per-run JSON + aggregated analysis
```

---

## 5. FAQ

- **Q: I see `Could not locate project root` from `figures/table_utils.py`.**
  A: Make sure you run the figure scripts from the release root. The discovery looks for `pyproject.toml` and `results/` next to the project root.

- **Q: A run errors with `FileNotFoundError` on a `.hdf5`.**
  A: Re-check that all 9 D4RL files are placed under `data/gym_mujoco_v2/<env_full>/` with the exact filenames listed in §3. The release does not download data on your behalf.

- **Q: The benchmark table is missing some rows.**
  A: `figures/gen_tables_main.py` raises if any `(env, block)` is missing from `results/analysis/benchmark_audit.json` or `b1_b4_multiseed_summary.json`. Re-run the `dt_baselines` and `trajdeleter` stages, then `analyze`, then `figures`.

- **Q: How long does a full reproduction take?**
  A: ~30 GPU-hours total on a single 16 GB GPU. Most of the cost comes from `dt_baselines` and `selective`. Use `--stages` to scope to the assets you need.