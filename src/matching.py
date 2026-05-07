"""Nearest-neighbor matching for building non-member sets."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from .antmaze_utils import (
    normalize_antmaze_offline_state_mode,
    select_antmaze_offline_states,
)


def extract_trajectory_features(
    trajectories: list[dict],
    stats: dict,
    profile: str = "basic",
    antmaze_offline_state_mode: str = "observations",
) -> np.ndarray:
    features = []
    state_mean = stats["state_mean"]
    state_std = stats["state_std"]
    profile_name = str(profile).strip().lower()
    offline_state_mode = normalize_antmaze_offline_state_mode(
        antmaze_offline_state_mode
    )

    for traj in trajectories:
        z_return = (traj["return"] - stats["return_mean"]) / stats["return_std"]
        z_length = (traj["length"] - stats["length_mean"]) / stats["length_std"]
        obs = select_antmaze_offline_states(traj, offline_state_mode)
        z_s0 = (obs[0] - state_mean) / state_std

        feat_parts = [[z_return, z_length], z_s0]
        if profile_name == "stronger":
            rewards = np.asarray(traj["rewards"], dtype=np.float32)
            actions = np.asarray(traj["actions"], dtype=np.float32)
            reward_mean = float(rewards.mean()) if rewards.size else 0.0
            reward_std = float(rewards.std()) if rewards.size else 0.0
            reward_last = float(rewards[-1]) if rewards.size else 0.0
            action_abs_mean = actions.mean(axis=0) if actions.size else np.zeros((0,))
            action_abs_mean = np.abs(action_abs_mean)
            action_abs_std = (
                np.abs(actions).std(axis=0) if actions.size else np.zeros((0,))
            )
            obs_delta = (
                obs[-1] - obs[0] if obs.shape[0] > 0 else np.zeros_like(state_mean)
            )
            obs_delta_norm = np.linalg.norm(obs_delta / state_std)
            feat_parts.extend(
                [
                    [reward_mean, reward_std, reward_last, float(obs_delta_norm)],
                    action_abs_mean,
                    action_abs_std,
                ]
            )

        feat = np.concatenate(feat_parts)
        features.append(feat)

    return np.array(features, dtype=np.float32)


def tune_matching_threshold(
    train_feats: np.ndarray,
    cal_feats: np.ndarray,
    forget_ratio: float = 0.1,
    seed: int = 42,
    percentile: float = 90.0,
    full_coverage: bool = False,
) -> float:
    """Tune matching threshold by simulating forget set from train.

    1. Sample forget_ratio of train as simulated forget
    2. Find nearest neighbor in calibration for each
    3. Threshold = percentile-th percentile of NN distances

    If full_coverage=True, use the maximum NN distance (ensures all
    simulated forget trajectories have at least one eligible candidate).
    """
    rng = np.random.RandomState(seed)
    n_sim = max(1, int(len(train_feats) * forget_ratio))
    sim_indices = rng.choice(len(train_feats), n_sim, replace=False)
    sim_feats = train_feats[sim_indices]

    tree = KDTree(cal_feats)
    distances, _ = tree.query(sim_feats, k=1)

    if full_coverage:
        threshold = float(np.max(distances)) * 1.1  # 10% headroom
        print(f"Matching threshold (full_coverage, max*1.1): {threshold:.4f}")
    else:
        threshold = np.percentile(distances, percentile)
        print(f"Matching threshold (p{percentile:.0f}): {threshold:.4f}")
    print(
        f"  Distance stats: mean={distances.mean():.4f}, "
        f"std={distances.std():.4f}, max={distances.max():.4f}"
    )
    return threshold


def build_matched_negative_set(
    forget_feats: np.ndarray,
    test_feats: np.ndarray,
    test_trajectories: list[dict],
    threshold: float,
) -> tuple[list[tuple[int, int]], dict]:
    n_forget = len(forget_feats)
    n_test = len(test_feats)
    if n_test != len(test_trajectories):
        raise ValueError("test_feats and test_trajectories have inconsistent counts")

    if n_forget == 0 or n_test == 0:
        quality_stats = {
            "n_forget": n_forget,
            "n_test": n_test,
            "n_matched": 0,
            "fraction_matched": 0.0,
            "threshold": float(threshold),
        }
        print(f"Matched 0/{n_forget} (0.0%)")
        return [], quality_stats

    distance_matrix = cdist(forget_feats, test_feats, metric="euclidean")
    eligible_mask = distance_matrix <= threshold
    eligible_counts = eligible_mask.sum(axis=1)
    nearest_distances = distance_matrix.min(axis=1)

    unmatched_penalty = float(threshold + max(1.0, threshold * 0.1))
    large_penalty = unmatched_penalty * 1000.0
    padded_cost = np.full(
        (n_forget, n_test + n_forget), large_penalty, dtype=np.float64
    )
    padded_cost[:, :n_test] = np.where(eligible_mask, distance_matrix, large_penalty)
    for idx in range(n_forget):
        padded_cost[idx, n_test + idx] = unmatched_penalty

    row_ind, col_ind = linear_sum_assignment(padded_cost)

    matched_pairs = []
    matched_distances = []
    strict_matched_pairs = []
    strict_matched_distances = []
    unmatched_due_to_threshold = 0
    unmatched_due_to_assignment = 0
    unmatched_forget_indices = []
    used_test_indices = set()

    for f_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
        if col_idx < n_test and eligible_mask[f_idx, col_idx]:
            matched_pairs.append((f_idx, int(col_idx)))
            matched_distances.append(float(distance_matrix[f_idx, col_idx]))
            strict_matched_pairs.append((f_idx, int(col_idx)))
            strict_matched_distances.append(float(distance_matrix[f_idx, col_idx]))
            used_test_indices.add(int(col_idx))
        elif eligible_counts[f_idx] == 0:
            unmatched_due_to_threshold += 1
            unmatched_forget_indices.append(f_idx)
        else:
            unmatched_due_to_assignment += 1
            unmatched_forget_indices.append(f_idx)

    fallback_pairs = []
    fallback_distances = []
    available_test_indices = sorted(set(range(n_test)) - used_test_indices)
    if unmatched_forget_indices and available_test_indices:
        fallback_cost = distance_matrix[
            np.ix_(unmatched_forget_indices, available_test_indices)
        ]
        fallback_row_ind, fallback_col_ind = linear_sum_assignment(fallback_cost)
        for row_idx, col_idx in zip(
            fallback_row_ind.tolist(), fallback_col_ind.tolist()
        ):
            forget_idx = int(unmatched_forget_indices[row_idx])
            test_idx = int(available_test_indices[col_idx])
            fallback_pairs.append((forget_idx, test_idx))
            fallback_distances.append(float(distance_matrix[forget_idx, test_idx]))

    matched_pairs.extend(fallback_pairs)
    matched_distances.extend(fallback_distances)

    if matched_pairs:
        feat_deltas = []
        matched_test_indices = []
        for f_idx, t_idx in matched_pairs:
            matched_test_indices.append(t_idx)
            feat_deltas.append(np.abs(forget_feats[f_idx] - test_feats[t_idx]))
        feat_deltas = np.array(feat_deltas)
        quality_stats = {
            "n_forget": n_forget,
            "n_test": n_test,
            "n_matched": len(matched_pairs),
            "fraction_matched": len(matched_pairs) / n_forget,
            "n_strict_matched": len(strict_matched_pairs),
            "strict_fraction_matched": len(strict_matched_pairs) / n_forget,
            "n_fallback_matched": len(fallback_pairs),
            "fallback_fraction_matched": len(fallback_pairs) / n_forget,
            "threshold": float(threshold),
            "distance_mean": float(np.mean(matched_distances)),
            "distance_std": float(np.std(matched_distances)),
            "distance_max": float(np.max(matched_distances)),
            "strict_distance_mean": float(np.mean(strict_matched_distances))
            if strict_matched_distances
            else 0.0,
            "strict_distance_std": float(np.std(strict_matched_distances))
            if strict_matched_distances
            else 0.0,
            "strict_distance_max": float(np.max(strict_matched_distances))
            if strict_matched_distances
            else 0.0,
            "fallback_distance_mean": float(np.mean(fallback_distances))
            if fallback_distances
            else 0.0,
            "fallback_distance_std": float(np.std(fallback_distances))
            if fallback_distances
            else 0.0,
            "fallback_distance_max": float(np.max(fallback_distances))
            if fallback_distances
            else 0.0,
            "nearest_distance_mean": float(np.mean(nearest_distances)),
            "nearest_distance_std": float(np.std(nearest_distances)),
            "nearest_distance_max": float(np.max(nearest_distances)),
            "eligible_count_mean": float(np.mean(eligible_counts)),
            "eligible_count_min": int(np.min(eligible_counts)),
            "eligible_count_max": int(np.max(eligible_counts)),
            "n_with_zero_eligible": int(np.sum(eligible_counts == 0)),
            "unmatched_due_to_threshold": int(unmatched_due_to_threshold),
            "unmatched_due_to_assignment": int(unmatched_due_to_assignment),
            "fallback_used": bool(fallback_pairs),
            "matched_test_unique": int(len(set(matched_test_indices))),
            "per_feature_delta_mean": feat_deltas.mean(axis=0).tolist(),
        }
        if feat_deltas.shape[1] >= 1:
            quality_stats["return_delta_mean"] = float(feat_deltas[:, 0].mean())
        if feat_deltas.shape[1] >= 2:
            quality_stats["length_delta_mean"] = float(feat_deltas[:, 1].mean())
    else:
        quality_stats = {
            "n_forget": n_forget,
            "n_test": n_test,
            "n_matched": 0,
            "fraction_matched": 0.0,
            "n_strict_matched": 0,
            "strict_fraction_matched": 0.0,
            "n_fallback_matched": 0,
            "fallback_fraction_matched": 0.0,
            "threshold": float(threshold),
            "strict_distance_mean": 0.0,
            "strict_distance_std": 0.0,
            "strict_distance_max": 0.0,
            "fallback_distance_mean": 0.0,
            "fallback_distance_std": 0.0,
            "fallback_distance_max": 0.0,
            "nearest_distance_mean": float(np.mean(nearest_distances)),
            "nearest_distance_std": float(np.std(nearest_distances)),
            "nearest_distance_max": float(np.max(nearest_distances)),
            "eligible_count_mean": float(np.mean(eligible_counts)),
            "eligible_count_min": int(np.min(eligible_counts)),
            "eligible_count_max": int(np.max(eligible_counts)),
            "n_with_zero_eligible": int(np.sum(eligible_counts == 0)),
            "unmatched_due_to_threshold": int(unmatched_due_to_threshold),
            "unmatched_due_to_assignment": int(unmatched_due_to_assignment),
            "fallback_used": False,
        }

    print(
        f"Matched {len(matched_pairs)}/{len(forget_feats)} "
        f"({quality_stats['fraction_matched'] * 100:.1f}%)"
    )
    return matched_pairs, quality_stats
