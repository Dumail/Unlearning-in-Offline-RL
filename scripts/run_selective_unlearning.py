"""Component-Selective Unlearning

Usage:
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=attn
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=mlp
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=all
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=layer_2
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=attn.layer_1
    uv run python scripts/run_selective_unlearning.py env=hopper_mr seed=0 +target=scan_attn_layers
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    load_splits,
    parse_forget_ratio_cli_arg,
    resolve_ratio_artifact_dir,
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
    SelectiveGradientAscentUnlearner,
    build_valid_selective_targets,
)


DEFAULT_FORGET_RATIO = 0.10
CLI_FORGET_RATIO = parse_forget_ratio_cli_arg(
    sys.argv[1:], default=DEFAULT_FORGET_RATIO
)


def create_model(cfg, device):
    return model_factory_create_model(
        cfg,
        obs_dim=cfg.env.state_dim,
        act_dim=cfg.env.act_dim,
    ).to(device)


def resolve_targets(target: str, n_layers: int) -> list[str]:
    if target == "scan_layers":
        return [f"layer_{idx}" for idx in range(n_layers)]
    if target == "scan_attn_layers":
        return [f"attn.layer_{idx}" for idx in range(n_layers)]
    if target == "scan_mlp_layers":
        return [f"mlp.layer_{idx}" for idx in range(n_layers)]

    valid_targets = build_valid_selective_targets(n_layers)
    if target not in valid_targets:
        raise ValueError(
            f"Unknown target={target!r}, valid options: {sorted(valid_targets)}"
        )
    return [target]


def build_execution_targets(target: str, n_layers: int) -> list[str]:
    return resolve_targets(target, n_layers)


def _run_one_target(
    cfg,
    ckpt,
    target,
    forget_dataset,
    retain_dataset,
    retain_trajs,
    forget_trajs,
    negative_trajs,
    stats,
    matching_quality,
    device,
    mask_seed: int = 42,
):
    """Run one selective GA+Refit and return a results dict."""
    env_name = cfg.env.env_name
    model = create_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    base_model = create_model(cfg, device)
    base_model.load_state_dict(ckpt["model_state_dict"])

    unlearner = SelectiveGradientAscentUnlearner(
        model=model,
        base_model=base_model,
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
        target_components=target,
        kl_weight=cfg.unlearn.kl_weight,
        lr=cfg.unlearn.ascent_lr,
        grad_clip=cfg.unlearn.grad_clip,
        device=device,
        random_seed=mask_seed,
    )

    ascent_log = unlearner.run_ascent(cfg.unlearn.ascent_steps)
    refit_log = unlearner.refit_head(
        n_steps=cfg.unlearn.refit_steps,
        lr=cfg.unlearn.refit_lr,
        reinit_head=bool(getattr(cfg.unlearn, "reinit_refit_head", True)),
    )

    for p in model.parameters():
        p.requires_grad = True

    d4rl_score = evaluate(
        model,
        env_name,
        target_return=cfg.env.target_return,
        random_score=cfg.env.random_score,
        expert_score=cfg.env.expert_score,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        device=device,
    )

    tmi = full_tmi_evaluation(
        model,
        forget_trajs,
        negative_trajs,
        retain_trajs,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        context_length=cfg.train.context_length,
        device=device,
    )
    retain_fields = build_b4_retain_fields(
        tmi,
        load_base_retain_nll_mean(cfg.results_dir, env_name),
    )

    del model, base_model, unlearner
    torch.cuda.empty_cache()

    return {
        "target": target,
        "d4rl_score": d4rl_score,
        "refit_reinit_head": bool(getattr(cfg.unlearn, "reinit_refit_head", True)),
        "ascent_final_nll": ascent_log[-1]["forget_nll"],
        "ascent_final_kl": ascent_log[-1]["kl"],
        "refit_final_loss": refit_log[-1]["refit_loss"],
        **{k: v for k, v in tmi.items() if not isinstance(v, list)},
        **retain_fields,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = cfg.env.env_name
    seed = cfg.seed
    forget_ratio = float(getattr(cfg, "forget_ratio", CLI_FORGET_RATIO))
    target = str(getattr(cfg, "target", "attn")).strip().lower()
    ascent_steps = cfg.unlearn.ascent_steps
    mask_seed = int(getattr(cfg, "mask_seed", 42))

    base_ckpt = getattr(cfg, "base_ckpt", None) or str(
        Path(cfg.checkpoint_dir) / "dt_final.pt"
    )

    print(
        f"=== Selective Unlearning: {env_name}, seed={seed}, "
        f"target={target}, steps={ascent_steps} ==="
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt = torch.load(base_ckpt, map_location=device, weights_only=True)
    overrides = infer_dt_model_overrides_from_state_dict(ckpt["model_state_dict"])
    apply_dt_model_overrides(cfg, overrides)

    # Data
    data_dir = resolve_ratio_artifact_dir(Path(cfg.data_dir) / env_name, forget_ratio)
    splits, stats = load_splits(data_dir)
    forget_trajs_all = splits["forget"]
    retain_trajs = splits["retain"]

    forget_dataset = TrajectoryDataset(
        forget_trajs_all,
        context_length=cfg.train.context_length,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    )
    retain_dataset = TrajectoryDataset(
        retain_trajs,
        context_length=cfg.train.context_length,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    )
    forget_trajs, negative_trajs = load_matched_sets(data_dir, splits)
    matching_quality = load_matching_quality(data_dir)

    model_probe = create_model(cfg, device)
    n_layers = len(model_probe.blocks)
    del model_probe

    # Run specified target
    targets = build_execution_targets(target, n_layers)

    results_by_target = {}
    for t in targets:
        print(f"\n{'=' * 60}")
        print(f"=== Running target={t} ===")
        print(f"{'=' * 60}")
        r = _run_one_target(
            cfg,
            ckpt,
            t,
            forget_dataset,
            retain_dataset,
            retain_trajs,
            forget_trajs,
            negative_trajs,
            stats,
            matching_quality,
            device,
            mask_seed=mask_seed,
        )
        results_by_target[t] = r

    # Save results
    tag_target = target.replace(".", "-")
    mask_tag = f"_mask{mask_seed}" if mask_seed != 42 else ""
    tag = f"selective_{tag_target}_lambda{cfg.unlearn.kl_weight}_steps{ascent_steps}_seed{seed}{mask_tag}"
    if not bool(getattr(cfg.unlearn, "reinit_refit_head", True)):
        tag = f"{tag}_noreinit"
    results_dir = Path(cfg.results_dir) / "selective" / env_name
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "component_selective_unlearning",
        "env": env_name,
        "seed": seed,
        "mask_seed": mask_seed,
        "forget_ratio": forget_ratio,
        "kl_weight": float(cfg.unlearn.kl_weight),
        "ascent_steps": int(ascent_steps),
        "refit_reinit_head": bool(getattr(cfg.unlearn, "reinit_refit_head", True)),
        "results": results_by_target,
        **matching_quality,
    }

    out_path = results_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")

    if target != "all" and target in results_by_target and "all" in results_by_target:
        sel = results_by_target[target]
        uni = results_by_target["all"]

        print(f"\n{'=' * 60}")
        print(f"=== {target} vs all (uniform) comparison ===")
        print(f"{'=' * 60}")
        print(f"{'Metric':<25} {target:>12} {'all':>12} {'Delta':>10}")
        print(f"{'-' * 60}")

        s_auc = sel["forget_auc"]
        u_auc = uni["forget_auc"]
        print(
            f"{'Forget AUC':<25} {s_auc:>12.4f} {u_auc:>12.4f} {s_auc - u_auc:>+10.4f}"
        )

        s_ret = sel["retain_diag_auc"]
        u_ret = uni["retain_diag_auc"]
        print(
            f"{'Retain Diag AUC':<25} {s_ret:>12.4f} {u_ret:>12.4f} {s_ret - u_ret:>+10.4f}"
        )

        s_d = sel["d4rl_score"]
        u_d = uni["d4rl_score"]
        print(f"{'D4RL Score':<25} {s_d:>12.2f} {u_d:>12.2f} {s_d - u_d:>+10.2f}")

        s_g = sel["gold_standard_valid"]
        u_g = uni["gold_standard_valid"]
        print(f"{'Gold Standard Valid':<25} {str(s_g):>12} {str(u_g):>12}")

        s_priv = abs(s_auc - 0.5)
        u_priv = abs(u_auc - 0.5)
        privacy_better = s_priv < u_priv
        utility_better = s_d > u_d
        print(
            f"Privacy erasure (AUC->0.5): {'selective better' if privacy_better else 'uniform better'}"
        )
        print(
            f"Utility retention (D4RL up): {'selective better' if utility_better else 'uniform better'}"
        )
        if privacy_better and utility_better:
            print(f">>> {target}-only GA Pareto-dominates uniform!")
        elif privacy_better or utility_better:
            print(f">>> {target}-only GA improves on one dimension.")
        else:
            print(f">>> {target}-only GA does not improve over uniform.")


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
