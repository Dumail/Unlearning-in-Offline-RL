from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .antmaze_utils import augment_states_with_goal_mode


@dataclass(frozen=True)
class EnvSpec:
    offline_name: str
    gym_name: str
    uses_qpos_qvel_state: bool = False


_ANTMAZE_SPECS: dict[str, EnvSpec] = {
    "antmaze-umaze-v2": EnvSpec(
        offline_name="antmaze-umaze-v2",
        gym_name="AntMaze_UMaze-v5",
        uses_qpos_qvel_state=True,
    ),
    "antmaze-umaze-diverse-v2": EnvSpec(
        offline_name="antmaze-umaze-diverse-v2",
        gym_name="AntMaze_UMaze-v5",
        uses_qpos_qvel_state=True,
    ),
    "antmaze-medium-play-v2": EnvSpec(
        offline_name="antmaze-medium-play-v2",
        gym_name="AntMaze_Medium-v5",
        uses_qpos_qvel_state=True,
    ),
    "antmaze-medium-diverse-v2": EnvSpec(
        offline_name="antmaze-medium-diverse-v2",
        gym_name="AntMaze_Medium-v5",
        uses_qpos_qvel_state=True,
    ),
    "antmaze-large-play-v2": EnvSpec(
        offline_name="antmaze-large-play-v2",
        gym_name="AntMaze_Large-v5",
        uses_qpos_qvel_state=True,
    ),
    "antmaze-large-diverse-v2": EnvSpec(
        offline_name="antmaze-large-diverse-v2",
        gym_name="AntMaze_Large-v5",
        uses_qpos_qvel_state=True,
    ),
}

_MUJOCO_ENV_MAP = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}


def ensure_env_dependencies(env_name: str) -> None:
    if not is_antmaze_env(env_name):
        return

    importlib.import_module("gymnasium_robotics")
    importlib.import_module("d4rl")


def get_env_spec(env_name: str) -> EnvSpec:
    env_name = str(env_name)
    ensure_env_dependencies(env_name)
    antmaze_spec = _ANTMAZE_SPECS.get(env_name)
    if antmaze_spec is not None:
        return antmaze_spec

    base_env = env_name.split("-")[0]
    gym_name = _MUJOCO_ENV_MAP.get(base_env, env_name.replace("-v2", "-v4"))
    return EnvSpec(offline_name=env_name, gym_name=gym_name)


def is_antmaze_env(env_name: str) -> bool:
    return str(env_name).startswith("antmaze-")


def extract_qpos_qvel_state(env: Any) -> np.ndarray:
    base_env = env.unwrapped
    qpos = np.asarray(base_env.data.qpos, dtype=np.float32)
    qvel = np.asarray(base_env.data.qvel, dtype=np.float32)
    return np.concatenate([qpos, qvel], axis=0)


def extract_desired_goal(obs: Any) -> np.ndarray:
    if not isinstance(obs, dict):
        raise TypeError(
            "AntMaze goal mode requires the environment to return dict observations"
        )
    desired_goal = obs.get("desired_goal")
    if desired_goal is None:
        raise KeyError("Environment observation dict is missing the 'desired_goal' key")
    desired_goal_arr = np.asarray(desired_goal, dtype=np.float32)
    if desired_goal_arr.ndim != 1:
        raise ValueError("AntMaze desired_goal must be a 1D vector")
    return desired_goal_arr


def append_goal_to_state(state: np.ndarray, obs: Any, goal_mode: str) -> np.ndarray:
    goal = extract_desired_goal(obs)
    return augment_states_with_goal_mode(
        np.asarray(state, dtype=np.float32)[None, :],
        goal[None, :],
        goal_mode,
    )[0]


def extract_success_flag(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    return bool(info.get("success", info.get("is_success", False)))


def score_rollouts(
    env_name: str,
    returns: list[float],
    random_score: float,
    expert_score: float,
    successes: list[bool] | None = None,
) -> float:
    if is_antmaze_env(env_name) and successes is not None and len(successes) > 0:
        return 100.0 * float(np.mean(np.asarray(successes, dtype=np.float32)))

    mean_return = float(np.mean(np.asarray(returns, dtype=np.float32)))
    return normalize_eval_score(mean_return, random_score, expert_score)


def extract_state_from_env_observation(env: Any, obs: Any) -> np.ndarray:
    spec = get_env_spec(getattr(getattr(env, "spec", None), "id", ""))
    if not spec.uses_qpos_qvel_state and isinstance(obs, dict):
        observation = obs.get("observation")
        if observation is None:
            raise KeyError(
                "Environment observation dict is missing the 'observation' key"
            )
        return np.asarray(observation, dtype=np.float32)

    if spec.uses_qpos_qvel_state:
        return extract_qpos_qvel_state(env)

    return np.asarray(obs, dtype=np.float32)


def extract_state_for_env_name(
    env_name: str,
    env: Any,
    obs: Any,
    antmaze_state_mode: str = "qpos_qvel",
    prev_qpos_qvel_state: np.ndarray | None = None,
) -> np.ndarray:
    spec = get_env_spec(env_name)
    if spec.uses_qpos_qvel_state:
        mode = str(antmaze_state_mode).strip().lower()
        if mode == "qpos_qvel":
            return extract_qpos_qvel_state(env)
        if mode == "prev_qpos_qvel":
            if prev_qpos_qvel_state is not None:
                return np.asarray(prev_qpos_qvel_state, dtype=np.float32)
            return extract_qpos_qvel_state(env)
        if mode == "observation_first29":
            if not isinstance(obs, dict):
                raise TypeError(
                    "AntMaze observation_first29 mode requires the environment to return dict observations"
                )
            observation = obs.get("observation")
            if observation is None:
                raise KeyError(
                    "Environment observation dict is missing the 'observation' key"
                )
            observation_arr = np.asarray(observation, dtype=np.float32)
            if observation_arr.ndim != 1 or observation_arr.shape[0] < 29:
                raise ValueError(
                    "AntMaze observation_first29 mode requires observation to be at least 29-dimensional"
                )
            return observation_arr[:29]
        if mode == "qpos_qvel_goal":
            return append_goal_to_state(
                extract_qpos_qvel_state(env), obs, "append_goal"
            )
        if mode == "prev_qpos_qvel_goal":
            qpos_qvel_state = extract_qpos_qvel_state(env)
            base_state = (
                np.asarray(prev_qpos_qvel_state, dtype=np.float32)
                if prev_qpos_qvel_state is not None
                else qpos_qvel_state
            )
            return append_goal_to_state(base_state, obs, "append_goal")
        if mode == "observation_first29_goal":
            if not isinstance(obs, dict):
                raise TypeError(
                    "AntMaze observation_first29_goal mode requires the environment to return dict observations"
                )
            observation = obs.get("observation")
            if observation is None:
                raise KeyError(
                    "Environment observation dict is missing the 'observation' key"
                )
            observation_arr = np.asarray(observation, dtype=np.float32)
            if observation_arr.ndim != 1 or observation_arr.shape[0] < 29:
                raise ValueError(
                    "AntMaze observation_first29_goal mode requires observation to be at least 29-dimensional"
                )
            return append_goal_to_state(observation_arr[:29], obs, "append_goal")
        if mode == "qpos_qvel_relative_goal":
            return append_goal_to_state(
                extract_qpos_qvel_state(env), obs, "relative_goal"
            )
        if mode == "prev_qpos_qvel_relative_goal":
            qpos_qvel_state = extract_qpos_qvel_state(env)
            base_state = (
                np.asarray(prev_qpos_qvel_state, dtype=np.float32)
                if prev_qpos_qvel_state is not None
                else qpos_qvel_state
            )
            return append_goal_to_state(base_state, obs, "relative_goal")
        if mode == "observation_first29_relative_goal":
            if not isinstance(obs, dict):
                raise TypeError(
                    "AntMaze observation_first29_relative_goal mode requires the environment to return dict observations"
                )
            observation = obs.get("observation")
            if observation is None:
                raise KeyError(
                    "Environment observation dict is missing the 'observation' key"
                )
            observation_arr = np.asarray(observation, dtype=np.float32)
            if observation_arr.ndim != 1 or observation_arr.shape[0] < 29:
                raise ValueError(
                    "AntMaze observation_first29_relative_goal mode requires observation to be at least 29-dimensional"
                )
            return append_goal_to_state(
                observation_arr[:29],
                obs,
                "relative_goal",
            )
        raise ValueError(
            "Unknown AntMaze state extraction mode: "
            f"{antmaze_state_mode}. Valid options: qpos_qvel, prev_qpos_qvel, observation_first29, qpos_qvel_goal, prev_qpos_qvel_goal, observation_first29_goal, qpos_qvel_relative_goal, prev_qpos_qvel_relative_goal, observation_first29_relative_goal"
        )

    if isinstance(obs, dict):
        observation = obs.get("observation")
        if observation is None:
            raise KeyError(
                "Environment observation dict is missing the 'observation' key"
            )
        return np.asarray(observation, dtype=np.float32)

    return np.asarray(obs, dtype=np.float32)


def normalize_eval_score(
    mean_return: float, random_score: float, expert_score: float
) -> float:
    denom = float(expert_score) - float(random_score)
    if np.isclose(denom, 0.0):
        return float(mean_return)
    return 100.0 * (float(mean_return) - float(random_score)) / denom
