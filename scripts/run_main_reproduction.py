from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCOMOTION_ENVS = [
    ("hopper_mr", "hopper-medium-replay-v2"),
    ("halfcheetah_mr", "halfcheetah-medium-replay-v2"),
    ("walker2d_mr", "walker2d-medium-replay-v2"),
    ("hopper_m", "hopper-medium-v2"),
    ("halfcheetah_m", "halfcheetah-medium-v2"),
    ("walker2d_m", "walker2d-medium-v2"),
    ("hopper_me", "hopper-medium-expert-v2"),
    ("halfcheetah_me", "halfcheetah-medium-expert-v2"),
    ("walker2d_me", "walker2d-medium-expert-v2"),
]

REPLAY_ENVS = [
    ("hopper_mr", "hopper-medium-replay-v2"),
    ("halfcheetah_mr", "halfcheetah-medium-replay-v2"),
    ("walker2d_mr", "walker2d-medium-replay-v2"),
]

DEFAULT_SEEDS = [0, 1, 2]
SELECTIVE_TARGETS = ["all", "attn", "attn.layer_0", "attn.layer_1", "attn.layer_2"]
SELECTIVE_STEPS = [100, 250, 500]

ALL_STAGES = (
    "dt_baselines",
    "trajdeleter",
    "backbone",
    "selective",
    "analyze",
    "figures",
)


def run_cmd(cmd: list[str], dry_run: bool, cwd: Path = PROJECT_ROOT) -> int:
    """Execute a command; when dry_run is set, only print without executing."""
    pretty = " ".join(shlex.quote(arg) for arg in cmd)
    print(f"$ {pretty}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _hydra_cmd(script: str, *overrides: str) -> list[str]:
    return ["uv", "run", "python", f"scripts/{script}", *overrides]


# ---------- Stage: data prep ----------


def stage_data_prep(envs: list[tuple[str, str]], dry_run: bool) -> int:
    """Download/split data + build matched negative set. Run once before experiments."""
    rc = 0
    for env_short, _ in envs:
        rc |= run_cmd(_hydra_cmd("download_data.py", f"env={env_short}"), dry_run)
        rc |= run_cmd(_hydra_cmd("build_negative_set.py", f"env={env_short}"), dry_run)
    return rc


# ---------- Stage: dt_baselines (B1+B2+B3+B4+TMI multi-seed) ----------


def stage_dt_baselines(
    envs: list[tuple[str, str]], seeds: list[int], dry_run: bool
) -> int:
    """For each (env, seed) run B1 base DT + B2 gold + B3 naive FT + B4 GA+Refit with TMI re-evaluation.

    Canonical config:
      B1 ckpt path: checkpoints/base/<env_full>/seed_<seed>/dt_final.pt
      B4 GA+Refit: kl_weight=0.1, ascent_steps=500, refit_steps=10000
    """
    rc = 0
    for env_short, env_full in envs:
        for seed in seeds:
            base_ckpt_dir = f"checkpoints/base/{env_full}/seed_{seed}"
            base_ckpt_path = f"{base_ckpt_dir}/dt_final.pt"
            rc |= run_cmd(
                _hydra_cmd(
                    "train_base_dt.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    f"checkpoint_dir={base_ckpt_dir}",
                ),
                dry_run,
            )
            rc |= run_cmd(
                _hydra_cmd(
                    "evaluate_tmi.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    f"checkpoint_dir={base_ckpt_dir}",
                ),
                dry_run,
            )
            rc |= run_cmd(
                _hydra_cmd("run_gold_standard.py", f"env={env_short}", f"seed={seed}"),
                dry_run,
            )
            rc |= run_cmd(
                _hydra_cmd(
                    "run_naive_ft.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    f"+base_ckpt={base_ckpt_path}",
                ),
                dry_run,
            )
            rc |= run_cmd(
                _hydra_cmd(
                    "run_unlearning.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    f"+base_ckpt={base_ckpt_path}",
                    "unlearn.kl_weight=0.1",
                    "unlearn.ascent_steps=500",
                    "unlearn.refit_steps=10000",
                ),
                dry_run,
            )
    return rc


# ---------- Stage: trajdeleter ----------


def stage_trajdeleter(
    envs: list[tuple[str, str]], seeds: list[int], dry_run: bool
) -> int:
    """B4T canonical: alpha=1.0 beta=2.0 stage1_steps=100 stage2_steps=1000."""
    rc = 0
    for env_short, env_full in envs:
        for seed in seeds:
            base_ckpt_path = f"checkpoints/base/{env_full}/seed_{seed}/dt_final.pt"
            rc |= run_cmd(
                _hydra_cmd(
                    "run_trajdeleter_unlearning.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    "unlearn=trajdeleter",
                    f"+base_ckpt={base_ckpt_path}",
                    "unlearn.alpha=1.0",
                    "unlearn.beta=2.0",
                    "unlearn.stage1_steps=100",
                    "unlearn.stage2_steps=1000",
                ),
                dry_run,
            )
    return rc


# ---------- Stage: backbone (DT/MLP/LSTM matched-budget) ----------


def stage_backbone(envs: list[tuple[str, str]], seeds: list[int], dry_run: bool) -> int:
    """For each (env, seed) run DT+MLP+LSTM matched-budget training and evaluation via Hydra entry."""
    rc = 0
    for env_short, _ in envs:
        for seed in seeds:
            rc |= run_cmd(
                _hydra_cmd(
                    "run_backbone_comparison.py",
                    f"env={env_short}",
                    f"seed={seed}",
                    "+models=dt,mlp,lstm",
                ),
                dry_run,
            )
    return rc


# ---------- Stage: selective ----------


def stage_selective(
    envs: list[tuple[str, str]], seeds: list[int], dry_run: bool
) -> int:
    """5 targets × 3 ascent steps × 3 seeds × 3 replay envs。"""
    rc = 0
    for env_short, env_full in envs:
        for seed in seeds:
            base_ckpt_path = f"checkpoints/base/{env_full}/seed_{seed}/dt_final.pt"
            for target in SELECTIVE_TARGETS:
                for steps in SELECTIVE_STEPS:
                    rc |= run_cmd(
                        _hydra_cmd(
                            "run_selective_unlearning.py",
                            f"env={env_short}",
                            f"seed={seed}",
                            f"+base_ckpt={base_ckpt_path}",
                            f"+target={target}",
                            f"unlearn.ascent_steps={steps}",
                        ),
                        dry_run,
                    )
    return rc


# ---------- Stage: analyze ----------


def stage_analyze(dry_run: bool) -> int:
    """6 analysis scripts."""
    rc = 0
    rc |= run_cmd(_hydra_cmd("build_benchmark_audit.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("analyze_b1_b4_multiseed.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("analyze_multi_attack.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("analyze_b6_matched_budget.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("analyze_selective_results.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("run_layer_selection_cv.py"), dry_run)
    rc |= run_cmd(_hydra_cmd("analyze_selective_utility_budget.py"), dry_run)
    return rc


# ---------- Stage: figures ----------


def stage_figures(dry_run: bool) -> int:
    """Generate 4 main-text tables as plain text and 1 heatmap figure as PDF."""
    rc = 0
    rc |= run_cmd(
        ["uv", "run", "python", "figures/gen_tables_main.py"],
        dry_run,
        cwd=PROJECT_ROOT,
    )
    rc |= run_cmd(
        ["uv", "run", "python", "figures/gen_fig2_heatmap.py"],
        dry_run,
        cwd=PROJECT_ROOT,
    )
    return rc


# ---------- Main entry ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main-text minimal reproduction orchestrator"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(ALL_STAGES),
        choices=ALL_STAGES,
        help="Select stages to execute (run in the order provided).",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Override the default 9 locomotion env list (Hydra env names, e.g. hopper_mr).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seed list, default 0 1 2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands that would be executed, without actually running them.",
    )
    parser.add_argument(
        "--prep-data",
        action="store_true",
        help="Run download_data + build_negative_set before the stage sequence.",
    )
    return parser.parse_args()


def filter_envs(
    env_short_list: list[str] | None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if env_short_list is None:
        return LOCOMOTION_ENVS, REPLAY_ENVS
    selected = [(s, full) for (s, full) in LOCOMOTION_ENVS if s in env_short_list]
    selected_replay = [(s, full) for (s, full) in REPLAY_ENVS if s in env_short_list]
    return selected, selected_replay


def main() -> int:
    args = parse_args()
    locomotion_envs, replay_envs = filter_envs(args.envs)

    rc = 0
    if args.prep_data:
        print("\n=== Stage: data_prep ===")
        rc |= stage_data_prep(locomotion_envs, args.dry_run)

    for stage in args.stages:
        print(f"\n=== Stage: {stage} ===")
        if stage == "dt_baselines":
            rc |= stage_dt_baselines(locomotion_envs, args.seeds, args.dry_run)
        elif stage == "trajdeleter":
            rc |= stage_trajdeleter(locomotion_envs, args.seeds, args.dry_run)
        elif stage == "backbone":
            rc |= stage_backbone(replay_envs, args.seeds, args.dry_run)
        elif stage == "selective":
            rc |= stage_selective(replay_envs, args.seeds, args.dry_run)
        elif stage == "analyze":
            rc |= stage_analyze(args.dry_run)
        elif stage == "figures":
            rc |= stage_figures(args.dry_run)

    print(f"\n[orchestrator] return code: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
