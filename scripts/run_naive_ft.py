"""Naive Fine-Tuning: continue training base DT on retain set, evaluate TMI.

Usage:
    uv run python scripts/run_naive_ft.py                         # halfcheetah
    uv run python scripts/run_naive_ft.py env=hopper_mr           # hopper
    uv run python scripts/run_naive_ft.py seed=0                  # specific seed
    uv run python scripts/run_naive_ft.py +base_ckpt=path/to/dt_final.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    load_split_metadata,
    load_splits,
    parse_forget_ratio_cli_arg,
    resolve_ratio_artifact_dir,
    resolve_split_dir,
)
from src.antmaze_utils import (
    compute_augmented_state_stats,
    default_antmaze_state_mode,
    resolve_antmaze_goal_mode,
    resolve_antmaze_offline_state_mode,
    resolve_antmaze_reward_mode,
)
from src.dataset import TrajectoryDataset
from src.model_factory import (
    apply_dt_model_overrides,
    create_model as model_factory_create_model,
    infer_dt_model_overrides_from_state_dict,
)
from src.retain_checks import build_b4_retain_fields, load_base_retain_nll_mean
from src.tmi import full_tmi_evaluation, load_matched_sets, load_matching_quality
from src.trainer import DTTrainer, evaluate


DEFAULT_FORGET_RATIO = 0.10
CLI_FORGET_RATIO = parse_forget_ratio_cli_arg(
    sys.argv[1:], default=DEFAULT_FORGET_RATIO
)


def create_model(cfg, device):
    model = model_factory_create_model(
        cfg,
        obs_dim=cfg.env.state_dim,
        act_dim=cfg.env.act_dim,
    )
    if not hasattr(model, "to"):
        model_name = str(getattr(cfg, "model", "dt"))
        raise TypeError(
            f"Model '{model_name}' is not yet supported by run_naive_ft.py. "
            "This script currently supports DT-only until B6 is implemented."
        )
    return model.to(device)


def _metadata_str(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = cfg.env.env_name
    seed = cfg.seed
    forget_ratio = float(getattr(cfg, "forget_ratio", CLI_FORGET_RATIO))

    # Base checkpoint path
    base_ckpt = getattr(cfg, "base_ckpt", None) or str(
        Path(cfg.checkpoint_dir) / "dt_final.pt"
    )

    print(f"=== Naive Fine-Tuning: {env_name}, seed={seed} ===")
    print(f"Base checkpoint: {base_ckpt}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = torch.load(base_ckpt, map_location=device, weights_only=True)
    overrides = infer_dt_model_overrides_from_state_dict(ckpt["model_state_dict"])
    applied = apply_dt_model_overrides(cfg, overrides)
    print(
        "Using DT config from checkpoint state_dict: "
        f"embedding_dim={applied['embedding_dim']}, n_layers={applied['n_layers']}"
    )

    # Load data
    split_dir = resolve_split_dir(cfg.data_dir, env_name)
    data_dir = resolve_ratio_artifact_dir(split_dir, forget_ratio)
    splits, stats = load_splits(data_dir)
    split_metadata = load_split_metadata(data_dir)
    antmaze_goal_mode = resolve_antmaze_goal_mode(
        env_name,
        getattr(cfg, "antmaze_goal_mode", None),
    )
    antmaze_offline_state_mode = resolve_antmaze_offline_state_mode(
        env_name,
        getattr(cfg, "antmaze_offline_state_mode", None),
    )
    state_mean, state_std = compute_augmented_state_stats(
        splits["train"],
        antmaze_goal_mode,
        antmaze_offline_state_mode,
    )
    antmaze_reward_mode = resolve_antmaze_reward_mode(
        env_name,
        getattr(cfg, "antmaze_reward_mode", None),
    )
    antmaze_eval_backend = str(
        getattr(cfg, "antmaze_eval_backend", "gymnasium_v5")
    ).strip()
    adapter_name = _metadata_str(split_metadata, "adapter_name")
    antmaze_fixed_goal = None
    if adapter_name == "antmaze_fixed_goal_v1":
        if antmaze_eval_backend.lower() not in {"d4rl_v2", "d4rl", "v2"}:
            raise ValueError(
                "Detected fixed_goal_v1 AntMaze data adapter directory, but the current evaluation backend is not d4rl_v2. "
                "Please explicitly pass +antmaze_eval_backend=d4rl_v2."
            )
        goal_x = split_metadata.get("adapter_goal_x")
        goal_y = split_metadata.get("adapter_goal_y")
        if goal_x is None or goal_y is None:
            raise ValueError(
                "fixed_goal_v1 metadata is missing adapter_goal_x / adapter_goal_y, unable to set D4RL fixed evaluation goal."
            )
        antmaze_fixed_goal = np.asarray(
            [float(goal_x), float(goal_y)], dtype=np.float32
        )
    antmaze_state_mode = str(
        getattr(
            cfg,
            "antmaze_state_mode",
            default_antmaze_state_mode(
                antmaze_goal_mode,
                antmaze_offline_state_mode,
            ),
        )
    )
    with open_dict(cfg):
        cfg.env.state_dim = int(state_mean.shape[0])
        if not hasattr(cfg, "antmaze_state_mode"):
            cfg.antmaze_state_mode = antmaze_state_mode
        cfg.antmaze_eval_backend = antmaze_eval_backend
        if antmaze_fixed_goal is not None:
            cfg.antmaze_fixed_goal = antmaze_fixed_goal.tolist()
    retain_trajs = splits["retain"]
    skip_utility_eval = bool(getattr(cfg, "skip_utility_eval", False))
    print(f"Retain set: {len(retain_trajs)} trajectories")

    # Create dataset from retain set
    dataset = TrajectoryDataset(
        retain_trajs,
        context_length=cfg.train.context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )

    # Load base checkpoint and fine-tune on D_r
    model = create_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model from step {ckpt['step']}")

    ckpt_dir = Path(cfg.checkpoint_dir) / "naive_ft" / env_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = DTTrainer(model, dataset, cfg, device=device)
    trainer.train(
        checkpoint_dir=str(ckpt_dir),
        env_name=env_name,
        target_return=cfg.env.target_return,
        random_score=cfg.env.random_score,
        expert_score=cfg.env.expert_score,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_reward_mode=antmaze_reward_mode,
        antmaze_eval_backend=antmaze_eval_backend,
        antmaze_fixed_goal=antmaze_fixed_goal,
    )

    checkpoint_path = ckpt_dir / "dt_final.pt"

    # Evaluate D4RL utility
    d4rl_score: float | None = None
    if skip_utility_eval:
        print("\n[SKIP] Utility evaluation deferred for external reevaluation")
    else:
        d4rl_score = evaluate(
            model,
            env_name,
            target_return=cfg.env.target_return,
            random_score=cfg.env.random_score,
            expert_score=cfg.env.expert_score,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            antmaze_state_mode=antmaze_state_mode,
            antmaze_reward_mode=antmaze_reward_mode,
            antmaze_eval_backend=antmaze_eval_backend,
            antmaze_fixed_goal=antmaze_fixed_goal,
        )
        print(f"\nD4RL Score: {d4rl_score:.2f}")

    # TMI evaluation
    forget_trajs, negative_trajs = load_matched_sets(data_dir, splits)
    matching_quality = load_matching_quality(data_dir)
    tmi_results = full_tmi_evaluation(
        model,
        forget_trajs,
        negative_trajs,
        retain_trajs,
        state_mean=state_mean,
        state_std=state_std,
        context_length=cfg.train.context_length,
        device=device,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )
    retain_b4_fields = build_b4_retain_fields(
        tmi_results,
        load_base_retain_nll_mean(cfg.results_dir, env_name),
    )

    # Save results
    results = {
        "method": "naive_ft",
        "env": env_name,
        "seed": seed,
        "base_checkpoint": base_ckpt,
        "checkpoint_path": str(checkpoint_path),
        "d4rl_score": d4rl_score,
        "utility_eval_skipped": bool(skip_utility_eval),
        "utility_eval_pending": bool(skip_utility_eval),
        "antmaze_state_mode": antmaze_state_mode,
        "antmaze_reward_mode": antmaze_reward_mode,
        "antmaze_eval_backend": antmaze_eval_backend,
        "antmaze_fixed_goal": antmaze_fixed_goal.tolist()
        if antmaze_fixed_goal is not None
        else None,
        **matching_quality,
        **retain_b4_fields,
        **tmi_results,
    }

    results_dir = Path(cfg.results_dir) / env_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"naive_ft_seed{seed}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Gate check
    if tmi_results["forget_auc"] > 0.65:
        print(
            f"\n*** GATE PASSED: Naive FT AUC = {tmi_results['forget_auc']:.4f} > 0.65 ***"
        )
    else:
        print(
            f"\n*** WARNING: Naive FT AUC = {tmi_results['forget_auc']:.4f} < 0.65 ***"
        )
        print("  DT may not memorize trajectories strongly enough.")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
