from __future__ import annotations

from typing import Any

import numpy as np


def normalize_antmaze_goal_mode(goal_mode: str | None) -> str:
    mode = str(goal_mode or "none").strip().lower()
    if mode not in {"none", "append_goal", "relative_goal"}:
        raise ValueError(
            "Unknown AntMaze goal mode: "
            f"{goal_mode}. Valid options: none, append_goal, relative_goal"
        )
    return mode


def resolve_antmaze_goal_mode(env_name: str, goal_mode: str | None) -> str:
    mode = str(goal_mode or "").strip().lower()
    if mode:
        return normalize_antmaze_goal_mode(mode)
    if str(env_name).startswith("antmaze-"):
        return "relative_goal"
    return "none"


def normalize_antmaze_offline_state_mode(state_mode: str | None) -> str:
    mode = str(state_mode or "observations").strip().lower()
    if mode not in {"observations", "qpos_qvel"}:
        raise ValueError(
            "Unknown AntMaze offline state mode: "
            f"{state_mode}. Valid options: observations, qpos_qvel"
        )
    return mode


def normalize_antmaze_reward_mode(reward_mode: str | None) -> str:
    mode = str(reward_mode or "none").strip().lower()
    if mode not in {"none", "minus_one", "delayed"}:
        raise ValueError(
            "Unknown AntMaze reward mode: "
            f"{reward_mode}. Valid options: none, minus_one, delayed"
        )
    return mode


def resolve_antmaze_reward_mode(env_name: str, reward_mode: str | None) -> str:
    mode = str(reward_mode or "").strip().lower()
    if mode:
        return normalize_antmaze_reward_mode(mode)
    if str(env_name).startswith("antmaze-"):
        return "none"
    return "none"


def transform_antmaze_rewards(rewards: np.ndarray, reward_mode: str) -> np.ndarray:
    mode = normalize_antmaze_reward_mode(reward_mode)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    if mode == "minus_one":
        return rewards_arr - np.float32(1.0)
    if mode == "delayed":
        transformed = np.zeros_like(rewards_arr)
        if transformed.size > 0:
            transformed[-1] = rewards_arr.sum(dtype=np.float32)
        return transformed
    return rewards_arr


def resolve_antmaze_offline_state_mode(env_name: str, state_mode: str | None) -> str:
    mode = str(state_mode or "").strip().lower()
    if mode:
        return normalize_antmaze_offline_state_mode(mode)
    if str(env_name).startswith("antmaze-"):
        return "observations"
    return "observations"


def default_antmaze_state_mode(
    goal_mode: str, offline_state_mode: str = "qpos_qvel"
) -> str:
    mode = normalize_antmaze_goal_mode(goal_mode)
    base_mode = normalize_antmaze_offline_state_mode(offline_state_mode)
    if base_mode == "observations":
        if mode == "append_goal":
            return "observation_first29_goal"
        if mode == "relative_goal":
            return "observation_first29_relative_goal"
        return "observation_first29"
    if mode == "append_goal":
        return "qpos_qvel_goal"
    if mode == "relative_goal":
        return "qpos_qvel_relative_goal"
    return "qpos_qvel"


def select_antmaze_offline_states(
    trajectory: dict[str, Any], offline_state_mode: str
) -> np.ndarray:
    mode = normalize_antmaze_offline_state_mode(offline_state_mode)
    if mode == "observations":
        return np.asarray(trajectory["observations"], dtype=np.float32)
    return np.asarray(trajectory["qpos_qvel_states"], dtype=np.float32)


def augment_states_with_goal_mode(
    states: np.ndarray,
    goals: np.ndarray | None,
    goal_mode: str,
) -> np.ndarray:
    mode = normalize_antmaze_goal_mode(goal_mode)
    states_arr = np.asarray(states, dtype=np.float32)
    if mode == "none":
        return states_arr

    if goals is None:
        raise KeyError(f"{mode} mode requires trajectories to contain a 'goals' field")

    goals_arr = np.asarray(goals, dtype=np.float32)
    if goals_arr.ndim != states_arr.ndim:
        raise ValueError(
            "goals and states have inconsistent dimensions; cannot construct AntMaze goal features"
        )
    if states_arr.shape[0] != goals_arr.shape[0]:
        raise ValueError("goals and states have inconsistent temporal lengths")

    if mode == "append_goal":
        return np.concatenate([states_arr, goals_arr], axis=-1)

    goal_dim = int(goals_arr.shape[-1])
    if states_arr.shape[-1] < goal_dim:
        raise ValueError(
            "states dimension is smaller than goal dimension; cannot construct relative goal features"
        )
    relative_goal = goals_arr - states_arr[..., :goal_dim]
    return np.concatenate([states_arr, relative_goal], axis=-1)


def compute_augmented_state_stats(
    trajectories: list[dict[str, Any]],
    goal_mode: str,
    offline_state_mode: str = "observations",
) -> tuple[np.ndarray, np.ndarray]:
    augmented_states = [
        augment_states_with_goal_mode(
            select_antmaze_offline_states(traj, offline_state_mode),
            traj.get("goals"),
            goal_mode,
        )
        for traj in trajectories
    ]
    all_states = np.concatenate(augmented_states, axis=0)
    state_mean = all_states.mean(axis=0).astype(np.float32)
    state_std = all_states.std(axis=0).astype(np.float32) + 1e-6
    return state_mean, state_std
