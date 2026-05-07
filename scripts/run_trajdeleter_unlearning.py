import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict

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
from src.trainer import evaluate
from src.unlearning import (
    TrajDeleterUnlearner,
    build_reward_flipped_dataset,
    compute_pre_unlearning_localization,
)


DEFAULT_FORGET_RATIO = 0.10
CLI_FORGET_RATIO = parse_forget_ratio_cli_arg(
    sys.argv[1:], default=DEFAULT_FORGET_RATIO
)

DEFAULT_STAGE1_LR = 1.0e-4
DEFAULT_STAGE2_LR = 1.0e-4
DEFAULT_GRAD_CLIP = 0.25


def _format_tag_value(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _build_run_tag(cfg: DictConfig, seed: int) -> str:
    tag = (
        f"alpha{cfg.unlearn.alpha}_beta{cfg.unlearn.beta}_"
        f"s1{cfg.unlearn.stage1_steps}_s2{cfg.unlearn.stage2_steps}_seed{seed}"
    )
    suffixes: list[str] = []
    if float(cfg.unlearn.stage1_lr) != DEFAULT_STAGE1_LR:
        suffixes.append(f"s1lr{_format_tag_value(float(cfg.unlearn.stage1_lr))}")
    if float(cfg.unlearn.stage2_lr) != DEFAULT_STAGE2_LR:
        suffixes.append(f"s2lr{_format_tag_value(float(cfg.unlearn.stage2_lr))}")
    if float(cfg.unlearn.grad_clip) != DEFAULT_GRAD_CLIP:
        suffixes.append(f"clip{_format_tag_value(float(cfg.unlearn.grad_clip))}")
    if suffixes:
        tag = f"{tag}_{'_'.join(suffixes)}"
    return tag


def create_model(cfg, device):
    model = model_factory_create_model(
        cfg,
        obs_dim=cfg.env.state_dim,
        act_dim=cfg.env.act_dim,
    )
    if not hasattr(model, "to"):
        model_name = str(getattr(cfg, "model", "dt"))
        raise TypeError(
            f"Model '{model_name}' is not yet supported by run_trajdeleter_unlearning.py. "
            "This script currently supports DT-only."
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

    base_ckpt = getattr(cfg, "base_ckpt", None) or str(
        Path(cfg.checkpoint_dir) / "dt_final.pt"
    )

    print(
        f"=== TrajDeleter: {env_name}, seed={seed}, alpha={cfg.unlearn.alpha}, beta={cfg.unlearn.beta} ==="
    )
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
    forget_trajs = splits["forget"]
    retain_trajs = splits["retain"]
    skip_utility_eval = bool(getattr(cfg, "skip_utility_eval", False))
    print(f"Forget: {len(forget_trajs)}, Retain: {len(retain_trajs)}")

    forget_dataset = TrajectoryDataset(
        forget_trajs,
        context_length=cfg.train.context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    forget_dataset_flipped = build_reward_flipped_dataset(
        forget_trajs,
        context_length=cfg.train.context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    retain_dataset = TrajectoryDataset(
        retain_trajs,
        context_length=cfg.train.context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )

    model = create_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])

    base_model = create_model(cfg, device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model from step {ckpt['step']}")

    localization = compute_pre_unlearning_localization(
        model=base_model,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        batch_size=int(getattr(cfg.unlearn, "diagnostic_batch_size", 32)),
        max_batches=int(getattr(cfg.unlearn, "diagnostic_batches", 2)),
        device=device,
    )
    print(
        "Pre-unlearning localization top target: "
        f"{localization['top_target']} ({localization['metric']})"
    )

    unlearner = TrajDeleterUnlearner(
        model=model,
        base_model=base_model,
        forget_dataset_flipped=forget_dataset_flipped,
        retain_dataset=retain_dataset,
        alpha=cfg.unlearn.alpha,
        beta=cfg.unlearn.beta,
        stage1_lr=cfg.unlearn.stage1_lr,
        grad_clip=cfg.unlearn.grad_clip,
        batch_size=int(getattr(cfg.unlearn, "batch_size", 64)),
        device=device,
    )
    stage1_log = unlearner.run_stage1(int(cfg.unlearn.stage1_steps))
    stage2_log = unlearner.run_stage2(
        int(cfg.unlearn.stage2_steps),
        lr=float(cfg.unlearn.stage2_lr),
        batch_size=int(getattr(cfg.unlearn, "batch_size", 64)),
    )

    tag = _build_run_tag(cfg, seed)
    ckpt_dir = Path(cfg.checkpoint_dir) / "trajdeleter" / env_name / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": tag},
        ckpt_dir / "dt_trajdeleter.pt",
    )

    d4rl_score: float | None = None
    if skip_utility_eval:
        print("\n[SKIP] Utility evaluation deferred for external reevaluation")
    else:
        for p in model.parameters():
            p.requires_grad = True
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
            eval_seed=int(seed),
        )
        print(f"\nD4RL Score: {d4rl_score:.2f}")

    forget_eval_trajs, negative_trajs = load_matched_sets(data_dir, splits)
    matching_quality = load_matching_quality(data_dir)
    tmi_results = full_tmi_evaluation(
        model,
        forget_eval_trajs,
        negative_trajs,
        retain_trajs,
        state_mean=state_mean,
        state_std=state_std,
        context_length=cfg.train.context_length,
        device=device,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    retain_fields = build_b4_retain_fields(
        tmi_results,
        load_base_retain_nll_mean(cfg.results_dir, env_name),
    )

    results = {
        "method": "trajdeleter",
        "env": env_name,
        "seed": seed,
        "alpha": float(cfg.unlearn.alpha),
        "beta": float(cfg.unlearn.beta),
        "stage1_steps": int(cfg.unlearn.stage1_steps),
        "stage2_steps": int(cfg.unlearn.stage2_steps),
        "stage1_lr": float(cfg.unlearn.stage1_lr),
        "stage2_lr": float(cfg.unlearn.stage2_lr),
        "base_checkpoint": base_ckpt,
        "checkpoint_path": str(ckpt_dir / "dt_trajdeleter.pt"),
        "pre_unlearning_localization": localization,
        "d4rl_score": d4rl_score,
        "utility_eval_skipped": bool(skip_utility_eval),
        "utility_eval_pending": bool(skip_utility_eval),
        "antmaze_state_mode": antmaze_state_mode,
        "antmaze_reward_mode": antmaze_reward_mode,
        "antmaze_eval_backend": antmaze_eval_backend,
        "antmaze_fixed_goal": antmaze_fixed_goal.tolist()
        if antmaze_fixed_goal is not None
        else None,
        "stage1_final_loss": stage1_log[-1]["loss"],
        "stage1_final_retain_nll": stage1_log[-1]["retain_nll"],
        "stage1_final_forget_flipped_nll": stage1_log[-1]["forget_flipped_nll"],
        "stage2_final_loss": stage2_log[-1]["loss"],
        "stage2_final_retain_nll": stage2_log[-1]["retain_nll"],
        "stage2_final_anchor_kl": stage2_log[-1]["anchor_kl"],
        **matching_quality,
        **retain_fields,
        **tmi_results,
    }

    results_dir = Path(cfg.results_dir) / env_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"trajdeleter_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n=== Summary ===")
    print(
        "Method: TrajDeleter "
        f"(alpha={cfg.unlearn.alpha}, beta={cfg.unlearn.beta}, "
        f"s1={cfg.unlearn.stage1_steps}, s2={cfg.unlearn.stage2_steps})"
    )
    print(
        f"Forget AUC: {tmi_results['forget_auc']:.4f} "
        f"[{tmi_results['forget_auc_ci_low']:.4f}, {tmi_results['forget_auc_ci_high']:.4f}]"
    )
    print(f"Retain Diag AUC: {tmi_results['retain_diag_auc']:.4f}")
    if d4rl_score is None:
        print("D4RL Score: pending external reevaluation")
    else:
        print(f"D4RL Score: {d4rl_score:.2f}")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
