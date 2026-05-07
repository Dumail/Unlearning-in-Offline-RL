from __future__ import annotations

"""Trajectory Membership Inference (TMI) evaluation.

Computes AUC-ROC for distinguishing forget trajectories from a matched non-member
set using mean per-token NLL scores, with bootstrap confidence intervals.
"""

from pathlib import Path

import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .data_pipeline import (
    DEFAULT_FORGET_RATIO,
    load_split_metadata,
    resolve_matched_negative_path,
)
from .nll import compute_trajectory_nll


def compute_all_trajectory_nlls(
    model: nn.Module,
    trajectories: list[dict],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    context_length: int = 20,
    device: str = "cuda",
    label: str = "",
    antmaze_goal_mode: str = "none",
    antmaze_offline_state_mode: str = "observations",
    antmaze_reward_mode: str = "none",
) -> np.ndarray:
    """Compute mean per-token NLL for all trajectories.

    Returns array of shape (n_trajectories,).
    Lower NLL = model is more confident on this trajectory.
    """
    nlls = []
    n = len(trajectories)
    for i, traj in enumerate(trajectories):
        nll = compute_trajectory_nll(
            model,
            traj,
            state_mean,
            state_std,
            context_length=context_length,
            device=device,
            antmaze_goal_mode=antmaze_goal_mode,
            antmaze_offline_state_mode=antmaze_offline_state_mode,
            antmaze_reward_mode=antmaze_reward_mode,
        )
        nlls.append(nll)
        if label and (i + 1) % max(1, n // 5) == 0:
            print(f"  [{label}] {i + 1}/{n} trajectories done")
    return np.array(nlls)


def compute_tmi_auc(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
) -> float:
    """Compute TMI AUC-ROC.

    Attack: lower NLL → more likely a member (trained on this data).
    Labels: forget=1 (member), negative=0 (non-member).
    Score: -NLL (higher = more likely member).
    """
    labels = np.concatenate([np.ones(len(forget_nlls)), np.zeros(len(negative_nlls))])
    scores = np.concatenate([-forget_nlls, -negative_nlls])
    return float(roc_auc_score(labels, scores))


def bootstrap_auc_ci(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for TMI AUC-ROC.

    Returns (auc_mean, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    n_f = len(forget_nlls)
    n_n = len(negative_nlls)

    aucs = []
    for _ in range(n_bootstrap):
        f_idx = rng.choice(n_f, n_f, replace=True)
        n_idx = rng.choice(n_n, n_n, replace=True)
        try:
            auc = compute_tmi_auc(forget_nlls[f_idx], negative_nlls[n_idx])
            aucs.append(auc)
        except ValueError:
            # Can happen if all labels are the same after resampling
            continue

    aucs = np.array(aucs)
    ci_low = np.percentile(aucs, 100 * alpha / 2)
    ci_high = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(np.mean(aucs)), float(ci_low), float(ci_high)


def paired_bootstrap_auc_ci(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    forget_arr = np.asarray(forget_nlls, dtype=np.float64)
    negative_arr = np.asarray(negative_nlls, dtype=np.float64)
    if forget_arr.shape != negative_arr.shape:
        raise ValueError("paired bootstrap requires equal-length matched arrays")

    n_pairs = forget_arr.size
    if n_pairs == 0:
        return float("nan"), float("nan"), float("nan")
    if n_pairs == 1:
        auc = compute_tmi_auc(forget_arr, negative_arr)
        return auc, auc, auc

    rng = np.random.RandomState(seed)
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        pair_idx = rng.choice(n_pairs, n_pairs, replace=True)
        aucs[idx] = compute_tmi_auc(forget_arr[pair_idx], negative_arr[pair_idx])

    ci_low = np.percentile(aucs, 100 * alpha / 2)
    ci_high = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(np.mean(aucs)), float(ci_low), float(ci_high)


def hierarchical_bootstrap_mean_auc_ci(
    seed_pairs: list[tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    normalized: list[tuple[np.ndarray, np.ndarray]] = []
    observed_aucs: list[float] = []
    for forget_nlls, negative_nlls in seed_pairs:
        forget_arr = np.asarray(forget_nlls, dtype=np.float64)
        negative_arr = np.asarray(negative_nlls, dtype=np.float64)
        if forget_arr.shape != negative_arr.shape:
            raise ValueError(
                "hierarchical bootstrap requires equal-length matched arrays per seed"
            )
        if forget_arr.size == 0:
            continue
        normalized.append((forget_arr, negative_arr))
        observed_aucs.append(compute_tmi_auc(forget_arr, negative_arr))

    if not normalized:
        return float("nan"), float("nan"), float("nan")
    if len(normalized) == 1:
        return paired_bootstrap_auc_ci(
            normalized[0][0],
            normalized[0][1],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )

    rng = np.random.RandomState(seed)
    n_seeds = len(normalized)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        seed_idx = rng.choice(n_seeds, n_seeds, replace=True)
        sampled_aucs: list[float] = []
        for draw_pos, pair_pos in enumerate(seed_idx):
            forget_arr, negative_arr = normalized[int(pair_pos)]
            pair_idx = rng.choice(forget_arr.size, forget_arr.size, replace=True)
            sampled_aucs.append(
                compute_tmi_auc(forget_arr[pair_idx], negative_arr[pair_idx])
            )
        boot_means[idx] = float(np.mean(sampled_aucs))

    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(observed_aucs)), float(ci_low), float(ci_high)


def permutation_test_auc(
    forget_nlls: np.ndarray,
    negative_nlls: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test for H0: AUC = 0.5.

    Shuffles forget/negative labels and recomputes AUC each time.
    Returns p-value (fraction of permutation AUCs at least as extreme
    as the observed AUC, measured by |AUC - 0.5|).
    """
    observed_auc = compute_tmi_auc(forget_nlls, negative_nlls)
    observed_deviation = abs(observed_auc - 0.5)

    pooled = np.concatenate([forget_nlls, negative_nlls])
    n_f = len(forget_nlls)
    rng = np.random.RandomState(seed)

    count_extreme = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(pooled))
        perm_forget = pooled[perm[:n_f]]
        perm_negative = pooled[perm[n_f:]]
        try:
            perm_auc = compute_tmi_auc(perm_forget, perm_negative)
            if abs(perm_auc - 0.5) >= observed_deviation:
                count_extreme += 1
        except ValueError:
            continue

    return (count_extreme + 1) / (n_permutations + 1)


def _coerce_npz_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_matched_artifact(
    data_dir: str | Path,
    matching_variant: str = "basic",
) -> dict:
    data_dir = Path(data_dir)
    metadata = load_split_metadata(data_dir)
    forget_ratio = float(metadata.get("forget_ratio", DEFAULT_FORGET_RATIO))
    matched = np.load(
        resolve_matched_negative_path(
            data_dir,
            forget_ratio,
            matching_variant=matching_variant,
        ),
        allow_pickle=False,
    )
    return {key: _coerce_npz_value(matched[key]) for key in matched.files}


def load_matched_sets(
    data_dir: str | Path,
    splits: dict[str, list[dict]],
    matching_variant: str = "basic",
) -> tuple[list[dict], list[dict]]:
    """Load matched forget and negative trajectory sets.

    Returns (forget_trajs, negative_trajs) where each forget trajectory
    has a matched non-member trajectory from the test split.
    """
    matched = load_matched_artifact(data_dir, matching_variant=matching_variant)
    forget_indices = matched["forget_indices"]
    test_indices = matched["test_indices"]

    forget_trajs = [splits["forget"][i] for i in forget_indices]
    negative_trajs = [splits["test"][i] for i in test_indices]

    print(
        f"Loaded matched sets: {len(forget_trajs)} forget, "
        f"{len(negative_trajs)} negative"
    )
    return forget_trajs, negative_trajs


def load_matching_quality(
    data_dir: str | Path,
    matching_variant: str = "basic",
) -> dict:
    matched = load_matched_artifact(data_dir, matching_variant=matching_variant)
    quality = {}
    for key, value in matched.items():
        if not key.startswith("quality_"):
            continue
        if isinstance(value, np.ndarray):
            quality[key] = value.tolist()
        elif isinstance(value, np.generic):
            quality[key] = value.item()
        else:
            quality[key] = value
    return quality


def full_tmi_evaluation(
    model: nn.Module,
    forget_trajs: list[dict],
    negative_trajs: list[dict],
    retain_trajs: list[dict],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    context_length: int = 20,
    device: str = "cuda",
    n_bootstrap: int = 10000,
    max_retain_for_diagnostic: int = 100,
    antmaze_goal_mode: str = "none",
    antmaze_offline_state_mode: str = "observations",
    antmaze_reward_mode: str = "none",
) -> dict:
    """Complete TMI evaluation with all metrics.

    Returns dict with forget AUC, retain diagnostic AUC, bootstrap CIs,
    raw NLL scores, and Gold Standard validity check.
    """
    print("Computing trajectory NLLs...")

    forget_nlls = compute_all_trajectory_nlls(
        model,
        forget_trajs,
        state_mean,
        state_std,
        context_length,
        device,
        label="forget",
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    negative_nlls = compute_all_trajectory_nlls(
        model,
        negative_trajs,
        state_mean,
        state_std,
        context_length,
        device,
        label="negative",
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )

    # Subsample retain for diagnostic (full set is too slow)
    if len(retain_trajs) > max_retain_for_diagnostic:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(retain_trajs), max_retain_for_diagnostic, replace=False)
        retain_for_diag = [retain_trajs[i] for i in idx]
    else:
        retain_for_diag = retain_trajs

    retain_nlls = compute_all_trajectory_nlls(
        model,
        retain_for_diag,
        state_mean,
        state_std,
        context_length,
        device,
        label="retain",
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )

    # Forget vs Negative (primary TMI)
    forget_auc = compute_tmi_auc(forget_nlls, negative_nlls)
    forget_auc_mean, forget_ci_low, forget_ci_high = bootstrap_auc_ci(
        forget_nlls,
        negative_nlls,
        n_bootstrap=n_bootstrap,
    )
    ci_width = forget_ci_high - forget_ci_low

    # Retain vs Negative (diagnostic)
    retain_diag_auc = compute_tmi_auc(retain_nlls, negative_nlls)
    _, retain_ci_low, retain_ci_high = bootstrap_auc_ci(
        retain_nlls,
        negative_nlls,
        n_bootstrap=n_bootstrap,
    )

    # Gold Standard validity check (permutation test)
    pvalue = permutation_test_auc(forget_nlls, negative_nlls)
    gold_valid = pvalue > 0.05

    result = {
        "forget_nlls": forget_nlls.tolist(),
        "negative_nlls": negative_nlls.tolist(),
        "retain_nlls": retain_nlls.tolist(),
        "forget_auc": forget_auc,
        "forget_auc_bootstrap_mean": forget_auc_mean,
        "forget_auc_ci_low": forget_ci_low,
        "forget_auc_ci_high": forget_ci_high,
        "forget_auc_ci_width": ci_width,
        "retain_diag_auc": retain_diag_auc,
        "retain_diag_ci_low": retain_ci_low,
        "retain_diag_ci_high": retain_ci_high,
        "gold_standard_valid": gold_valid,
        "forget_auc_pvalue": pvalue,
        "forget_nll_mean": float(forget_nlls.mean()),
        "forget_nll_std": float(forget_nlls.std()),
        "negative_nll_mean": float(negative_nlls.mean()),
        "negative_nll_std": float(negative_nlls.std()),
        "retain_nll_mean": float(retain_nlls.mean()),
        "retain_nll_std": float(retain_nlls.std()),
    }

    # Print report
    print(f"\n{'=' * 50}")
    print(f"TMI Evaluation Report")
    print(f"{'=' * 50}")
    print(f"Forget  NLL: {forget_nlls.mean():.4f} +/- {forget_nlls.std():.4f}")
    print(f"Negative NLL: {negative_nlls.mean():.4f} +/- {negative_nlls.std():.4f}")
    print(f"Retain  NLL: {retain_nlls.mean():.4f} +/- {retain_nlls.std():.4f}")
    print(
        f"\nForget TMI AUC: {forget_auc:.4f} "
        f"[{forget_ci_low:.4f}, {forget_ci_high:.4f}] "
        f"(CI width: {ci_width:.4f})"
    )
    print(
        f"Retain Diag AUC: {retain_diag_auc:.4f} "
        f"[{retain_ci_low:.4f}, {retain_ci_high:.4f}]"
    )
    print(f"Gold Standard Valid: {gold_valid} (permutation p={pvalue:.4f})")
    print(f"{'=' * 50}")

    return result
