"""Standalone TMI evaluation for any checkpoint.

Usage:
    uv run python scripts/evaluate_tmi.py +ckpt=checkpoints/dt_final.pt
    uv run python scripts/evaluate_tmi.py +ckpt=checkpoints/gold_standard/halfcheetah-medium-replay-v2/seed_42/dt_final.pt
    uv run python scripts/evaluate_tmi.py +ckpt=path env=hopper_mr
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    load_splits,
    load_split_metadata,
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
from src.experiment_contract import infer_tmi_method_label
from src.model_factory import (
    apply_dt_model_overrides,
    create_model as model_factory_create_model,
    infer_dt_model_overrides_from_state_dict,
)
from src.tmi import full_tmi_evaluation, load_matched_sets, load_matching_quality
from src.trainer import evaluate


DEFAULT_FORGET_RATIO = 0.10
CLI_FORGET_RATIO = parse_forget_ratio_cli_arg(
    sys.argv[1:], default=DEFAULT_FORGET_RATIO
)
DEFAULT_MATCHING_VARIANT = "basic"


def _canonical_output_name(ckpt_path: str, seed: int, ckpt_stem: str) -> str:
    method = infer_tmi_method_label(ckpt_path)
    if method == "base":
        if ckpt_stem != "dt_final":
            return f"tmi_eval_{ckpt_stem}.json"
        return (
            "tmi_eval_dt_final.json"
            if int(seed) == 0
            else f"tmi_eval_dt_seed{int(seed)}.json"
        )
    if method == "gold_standard":
        return f"gold_standard_seed{int(seed)}.json"
    if method == "naive_ft":
        return f"naive_ft_seed{int(seed)}.json"
    if method == "ga_refit":
        parent_name = Path(ckpt_path).resolve().parent.name
        if parent_name.startswith("lambda") and "_seed" in parent_name:
            return f"ga_refit_{parent_name}.json"
        if ckpt_stem.startswith("ga_refit_"):
            return f"{ckpt_stem}.json"
        return f"tmi_eval_{ckpt_stem}.json"
    if method == "trajdeleter":
        parent_name = Path(ckpt_path).resolve().parent.name
        if parent_name.startswith("alpha") and "_seed" in parent_name:
            return f"trajdeleter_{parent_name}.json"
        if ckpt_stem.startswith("trajdeleter_"):
            return f"{ckpt_stem}.json"
        return f"tmi_eval_{ckpt_stem}.json"
    return f"tmi_eval_{ckpt_stem}.json"


def _matching_suffix(matching_variant: str) -> str:
    variant = str(matching_variant).strip().lower()
    return "" if variant in {"", "basic"} else f"_{variant}"


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
    forget_ratio = float(getattr(cfg, "forget_ratio", CLI_FORGET_RATIO))
    matching_variant = (
        str(getattr(cfg, "matching_variant", DEFAULT_MATCHING_VARIANT)).strip().lower()
    )
    antmaze_eval_backend = str(
        getattr(cfg, "antmaze_eval_backend", "gymnasium_v5")
    ).strip()

    ckpt_path = getattr(cfg, "ckpt", None)
    if ckpt_path is None:
        ckpt_path = str(Path(cfg.checkpoint_dir) / "dt_final.pt")

    print(f"=== TMI Evaluation: {env_name} ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Matching variant: {matching_variant}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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
    antmaze_reward_mode = resolve_antmaze_reward_mode(
        env_name,
        getattr(cfg, "antmaze_reward_mode", None),
    )
    adapter_name = _metadata_str(split_metadata, "adapter_name")
    antmaze_fixed_goal: np.ndarray | None = None
    if adapter_name == "antmaze_fixed_goal_v1":
        if antmaze_eval_backend not in {"d4rl_v2", "d4rl", "v2"}:
            raise ValueError(
                "Detected fixed_goal_v1 data directory, but the current antmaze_eval_backend is not d4rl_v2. "
                "Please explicitly pass +antmaze_eval_backend=d4rl_v2."
            )
        goal_x = split_metadata.get("adapter_goal_x")
        goal_y = split_metadata.get("adapter_goal_y")
        if goal_x is None or goal_y is None:
            raise ValueError(
                "fixed_goal_v1 metadata is missing adapter_goal_x / adapter_goal_y."
            )
        antmaze_fixed_goal = np.asarray(
            [float(goal_x), float(goal_y)], dtype=np.float32
        )
    state_mean, state_std = compute_augmented_state_stats(
        splits["train"],
        antmaze_goal_mode,
        antmaze_offline_state_mode,
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

    model = model_factory_create_model(
        cfg,
        obs_dim=cfg.env.state_dim,
        act_dim=cfg.env.act_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # D4RL evaluation
    d4rl_score = evaluate(
        model,
        env_name,
        n_episodes=int(getattr(cfg, "eval_episodes", 100)),
        target_return=cfg.env.target_return,
        random_score=cfg.env.random_score,
        expert_score=cfg.env.expert_score,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_reward_mode=antmaze_reward_mode,
        device=device,
        antmaze_state_mode=antmaze_state_mode,
        antmaze_eval_backend=antmaze_eval_backend,
        antmaze_fixed_goal=antmaze_fixed_goal,
        eval_seed=int(cfg.seed),
    )
    print(f"D4RL Score: {d4rl_score:.2f}")

    # TMI evaluation
    if matching_variant == "unmatched":
        n_forget = len(splits["forget"])
        forget_trajs = list(splits["forget"])
        negative_trajs = list(splits["test"][:n_forget])
        matching_quality = {
            "quality_matching_variant": "unmatched",
            "quality_n_forget": n_forget,
            "quality_n_test": len(splits["test"]),
            "quality_n_matched": len(negative_trajs),
            "quality_fraction_matched": (
                len(negative_trajs) / n_forget if n_forget > 0 else 0.0
            ),
        }
    else:
        forget_trajs, negative_trajs = load_matched_sets(
            data_dir,
            splits,
            matching_variant=matching_variant,
        )
        matching_quality = load_matching_quality(
            data_dir,
            matching_variant=matching_variant,
        )
    tmi_results = full_tmi_evaluation(
        model,
        forget_trajs,
        negative_trajs,
        splits["retain"],
        state_mean=state_mean,
        state_std=state_std,
        context_length=cfg.train.context_length,
        device=device,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )

    # Save
    results = {
        "checkpoint": ckpt_path,
        "env": env_name,
        "seed": int(cfg.seed),
        "method": infer_tmi_method_label(ckpt_path),
        "d4rl_score": d4rl_score,
        "antmaze_state_mode": antmaze_state_mode,
        "antmaze_reward_mode": antmaze_reward_mode,
        "antmaze_eval_backend": antmaze_eval_backend,
        "antmaze_fixed_goal": (
            antmaze_fixed_goal.tolist() if antmaze_fixed_goal is not None else None
        ),
        **matching_quality,
        **tmi_results,
    }

    results_dir = Path(cfg.results_dir) / env_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(ckpt_path).stem
    output_name = _canonical_output_name(ckpt_path, int(cfg.seed), ckpt_stem)
    if matching_variant not in {"", "basic"}:
        stem = Path(output_name).stem
        suffix = Path(output_name).suffix
        output_name = f"{stem}{_matching_suffix(matching_variant)}{suffix}"
    results_path = results_dir / output_name
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
