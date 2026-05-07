"""Data pipeline: download D4RL HDF5, extract trajectories, split datasets."""

from __future__ import annotations

import os
import urllib.request
import importlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .antmaze_utils import transform_antmaze_rewards

h5py = importlib.import_module("h5py")


# D4RL GCS base URL
D4RL_BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/"

# Map env names to HDF5 filenames
ENV_TO_FILE = {
    "halfcheetah-medium-replay-v2": "halfcheetah_medium_replay-v2.hdf5",
    "hopper-medium-replay-v2": "hopper_medium_replay-v2.hdf5",
    "halfcheetah-medium-v2": "halfcheetah_medium-v2.hdf5",
    "hopper-medium-v2": "hopper_medium-v2.hdf5",
    "walker2d-medium-v2": "walker2d_medium-v2.hdf5",
    "halfcheetah-expert-v2": "halfcheetah_expert-v2.hdf5",
    "hopper-expert-v2": "hopper_expert-v2.hdf5",
    "walker2d-expert-v2": "walker2d_expert-v2.hdf5",
    "halfcheetah-medium-expert-v2": "halfcheetah_medium_expert-v2.hdf5",
    "hopper-medium-expert-v2": "hopper_medium_expert-v2.hdf5",
    "walker2d-medium-expert-v2": "walker2d_medium_expert-v2.hdf5",
    "walker2d-medium-replay-v2": "walker2d_medium_replay-v2.hdf5",
}

DEFAULT_FORGET_RATIO = 0.10


def _resolve_local_dataset_candidate(dataset_url: str, data_dir: str) -> Path | None:
    raw = str(dataset_url).strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        return None

    if parsed.scheme == "file":
        candidate = Path(parsed.path)
    else:
        candidate = Path(raw)

    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(Path.cwd() / candidate)
        candidates.append(Path(data_dir) / candidate)

    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def parse_forget_ratio_cli_arg(
    argv: list[str], default: float = DEFAULT_FORGET_RATIO
) -> float:
    for arg in list(argv):
        if not arg.startswith("forget_ratio="):
            continue
        raw = arg.split("=", 1)[1]
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"invalid forget_ratio override: {arg}") from exc
        argv.remove(arg)
        return value
    return default


def format_forget_ratio(forget_ratio: float) -> str:
    return f"{float(forget_ratio):.2f}"


def resolve_ratio_artifact_dir(
    base_dir: str | Path,
    forget_ratio: float,
    default_forget_ratio: float = DEFAULT_FORGET_RATIO,
) -> Path:
    base = Path(base_dir)
    ratio = float(forget_ratio)
    if np.isclose(ratio, default_forget_ratio):
        return base
    return base / f"ratio_f{format_forget_ratio(ratio)}"


def resolve_matched_negative_path(
    data_dir: str | Path,
    forget_ratio: float,
    default_forget_ratio: float = DEFAULT_FORGET_RATIO,
    matching_variant: str = "basic",
) -> Path:
    ratio_tag = format_forget_ratio(forget_ratio)
    variant = str(matching_variant).strip().lower()
    if np.isclose(float(forget_ratio), default_forget_ratio):
        filename = (
            "matched_negative_set.npz"
            if variant in {"", "basic"}
            else f"matched_negative_set_{variant}.npz"
        )
    else:
        filename = (
            f"matched_negative_set_f{ratio_tag}.npz"
            if variant in {"", "basic"}
            else f"matched_negative_set_f{ratio_tag}_{variant}.npz"
        )
    return Path(data_dir) / filename


def resolve_split_dir(data_dir: str | Path, env_name: str) -> Path:
    base = Path(data_dir)
    direct = base / "stats.npz"
    if direct.exists():
        return base
    return base / env_name


def load_split_metadata(save_dir: str | Path) -> dict:
    metadata_path = Path(save_dir) / "metadata.npz"
    if not metadata_path.exists():
        return {}
    raw = np.load(metadata_path, allow_pickle=False)
    metadata = {k: _coerce_metadata_value(raw[k]) for k in raw.files}
    return metadata


def _coerce_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return _coerce_metadata_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def require_forget_ratio_metadata(save_dir: str | Path) -> float:
    metadata = load_split_metadata(save_dir)
    if "forget_ratio" not in metadata:
        raise ValueError(
            "Missing ratio identifier: forget_ratio not found in metadata.npz."
        )
    return float(metadata["forget_ratio"])


def download_dataset(
    env_name: str,
    data_dir: str = "data",
    dataset_url: str | None = None,
) -> Path:
    """Download HDF5 dataset from D4RL GCS. Returns path to local file."""
    if dataset_url:
        local_candidate = _resolve_local_dataset_candidate(dataset_url, data_dir)
        if local_candidate is not None:
            print(f"Using local dataset: {local_candidate}")
            return local_candidate

        parsed = urlparse(str(dataset_url))
        if parsed.scheme in {"http", "https"}:
            filename = Path(parsed.path).name
            local_dir = Path(data_dir) / "external"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            if local_path.exists():
                print(f"Dataset already exists: {local_path}")
                return local_path
            print(f"Downloading {dataset_url} -> {local_path}")
            urllib.request.urlretrieve(str(dataset_url), local_path)
            print(
                f"Downloaded: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)"
            )
            return local_path

        raise FileNotFoundError(f"Local data file not found: {dataset_url}")

    if env_name not in ENV_TO_FILE:
        raise KeyError(
            f"env={env_name} has no registered default download mapping; please provide dataset_url in the env config."
        )

    filename = ENV_TO_FILE[env_name]
    local_dir = Path(data_dir) / "gym_mujoco_v2"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / filename

    if local_path.exists():
        print(f"Dataset already exists: {local_path}")
        return local_path

    url = D4RL_BASE_URL + filename
    print(f"Downloading {url} -> {local_path}")
    urllib.request.urlretrieve(url, local_path)
    print(f"Downloaded: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
    return local_path


def extract_trajectories(
    hdf5_path: str | Path,
    min_length: int = 10,
    boundary_mode: str = "terminals_or_timeouts",
    antmaze_reward_mode: str = "none",
) -> list[dict]:
    """Extract trajectories from D4RL HDF5 file.

    Segments flat arrays by terminals/timeouts flags. Filters out
    trajectories shorter than min_length steps.

    Returns list of dicts with keys:
        observations (T, obs_dim), actions (T, act_dim),
        rewards (T,), return (float), length (int)
    """
    with h5py.File(hdf5_path, "r") as f:
        observations = np.asarray(f["observations"][:], dtype=np.float32)
        actions = np.asarray(f["actions"][:], dtype=np.float32)
        rewards = np.asarray(f["rewards"][:], dtype=np.float32)
        terminals = np.asarray(f["terminals"][:])
        timeouts = np.asarray(f["timeouts"][:])
        goals = None
        qpos_qvel_states = None
        if "infos/goal" in f:
            goals = np.asarray(f["infos/goal"][:], dtype=np.float32)
        if "infos/qpos" in f and "infos/qvel" in f:
            qpos = np.asarray(f["infos/qpos"][:], dtype=np.float32)
            qvel = np.asarray(f["infos/qvel"][:], dtype=np.float32)
            qpos_qvel_states = np.concatenate([qpos, qvel], axis=1)

    mode = str(boundary_mode).strip().lower()
    if mode == "terminals_or_timeouts":
        done_flags = np.logical_or(terminals, timeouts)
    elif mode == "timeouts_only":
        done_flags = timeouts.astype(bool)
    else:
        raise ValueError(
            "boundary_mode only supports 'terminals_or_timeouts' or 'timeouts_only', "
            f"received: {boundary_mode}"
        )

    episode_ends = np.where(done_flags)[0]

    trajectories = []
    start = 0
    for end in episode_ends:
        length = end - start + 1
        if length >= min_length:
            traj = {
                "observations": observations[start : end + 1],
                "actions": actions[start : end + 1],
                "rewards": rewards[start : end + 1],
                "return": float(rewards[start : end + 1].sum()),
                "length": int(length),
            }
            if goals is not None:
                traj["goals"] = goals[start : end + 1]
            if qpos_qvel_states is not None:
                traj["qpos_qvel_states"] = qpos_qvel_states[start : end + 1]
            trajectories.append(traj)
        start = end + 1

    # Handle remaining data if last step wasn't a done
    if start < len(observations):
        length = len(observations) - start
        if length >= min_length:
            traj = {
                "observations": observations[start:],
                "actions": actions[start:],
                "rewards": rewards[start:],
                "return": float(rewards[start:].sum()),
                "length": int(length),
            }
            if goals is not None:
                traj["goals"] = goals[start:]
            if qpos_qvel_states is not None:
                traj["qpos_qvel_states"] = qpos_qvel_states[start:]
            trajectories.append(traj)

    print(
        "Extracted {} trajectories (min_length={}, boundary_mode={})".format(
            len(trajectories), min_length, mode
        )
    )
    return trajectories


def compute_dataset_stats(trajectories: list[dict]) -> dict:
    """Compute normalization statistics across all transitions.

    Returns dict with keys:
        state_mean (obs_dim,), state_std (obs_dim,),
        return_mean, return_std, length_mean, length_std
    """
    all_states = np.concatenate([t["observations"] for t in trajectories], axis=0)
    returns = np.array([t["return"] for t in trajectories])
    lengths = np.array([t["length"] for t in trajectories])

    stats = {
        "state_mean": all_states.mean(axis=0).astype(np.float32),
        "state_std": all_states.std(axis=0).astype(np.float32) + 1e-6,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()) + 1e-6,
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()) + 1e-6,
    }

    if trajectories and all("goals" in traj for traj in trajectories):
        all_goals = np.concatenate([t["goals"] for t in trajectories], axis=0)
        stats["goal_mean"] = all_goals.mean(axis=0).astype(np.float32)
        stats["goal_std"] = all_goals.std(axis=0).astype(np.float32) + 1e-6

    return stats


def filter_trajectories_by_return(
    trajectories: list[dict], min_return_exclusive: float
) -> list[dict]:
    threshold = float(min_return_exclusive)
    filtered = [traj for traj in trajectories if float(traj["return"]) > threshold]
    print(
        f"Filtered trajectories by return>{threshold:.4f}: "
        f"{len(filtered)}/{len(trajectories)} kept"
    )
    return filtered


def split_trajectories(
    trajectories: list[dict],
    ratios: tuple[float, ...] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split trajectories into train/calibration/test at trajectory level."""
    rng = np.random.RandomState(seed)
    n = len(trajectories)
    indices = rng.permutation(n)

    n_train = int(n * ratios[0])
    n_cal = int(n * ratios[1])

    train_idx = indices[:n_train]
    cal_idx = indices[n_train : n_train + n_cal]
    test_idx = indices[n_train + n_cal :]

    train = [trajectories[i] for i in train_idx]
    cal = [trajectories[i] for i in cal_idx]
    test = [trajectories[i] for i in test_idx]

    print(f"Split: train={len(train)}, cal={len(cal)}, test={len(test)}")
    return train, cal, test


def create_forget_retain(
    train: list[dict], forget_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split train into forget (10%) and retain (90%) sets."""
    rng = np.random.RandomState(seed)
    n = len(train)
    n_forget = max(1, int(n * forget_ratio))
    indices = rng.permutation(n)

    forget = [train[i] for i in indices[:n_forget]]
    retain = [train[i] for i in indices[n_forget:]]

    print(f"Forget/retain: forget={len(forget)}, retain={len(retain)}")
    return forget, retain


def save_splits(
    splits: dict[str, list[dict]],
    stats: dict,
    save_dir: str | Path,
    metadata: dict | None = None,
) -> None:
    """Save trajectory splits and stats to disk."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save stats
    np.savez(save_dir / "stats.npz", **stats)

    if metadata is not None:
        np.savez(save_dir / "metadata.npz", **metadata)

    # Save each split
    for name, trajs in splits.items():
        split_dir = save_dir / name
        split_dir.mkdir(exist_ok=True)
        for i, traj in enumerate(trajs):
            payload: dict[str, Any] = {
                "observations": traj["observations"],
                "actions": traj["actions"],
                "rewards": traj["rewards"],
                "ret": np.array(traj["return"]),
                "length": np.array(traj["length"]),
            }
            if "goals" in traj:
                payload["goals"] = traj["goals"]
            if "qpos_qvel_states" in traj:
                payload["qpos_qvel_states"] = traj["qpos_qvel_states"]
            np.savez_compressed(split_dir / f"traj_{i:05d}.npz", **payload)
        # Save index
        np.savez(
            split_dir / "index.npz",
            returns=np.array([t["return"] for t in trajs]),
            lengths=np.array([t["length"] for t in trajs]),
            n_trajectories=np.array(len(trajs)),
        )

    print(f"Saved splits to {save_dir}")


def load_splits(save_dir: str | Path) -> tuple[dict[str, list[dict]], dict]:
    """Load trajectory splits and stats from disk."""
    save_dir = Path(save_dir)

    # Load stats
    stats_data = np.load(save_dir / "stats.npz")
    stats = {k: stats_data[k] for k in stats_data.files}
    # Convert scalars
    for k in ["return_mean", "return_std", "length_mean", "length_std"]:
        if k in stats:
            stats[k] = float(stats[k])

    splits = {}
    for split_name in ["train", "calibration", "test", "forget", "retain"]:
        split_dir = save_dir / split_name
        if not split_dir.exists():
            continue

        index = np.load(split_dir / "index.npz")
        n = int(index["n_trajectories"])
        trajs = []
        for i in range(n):
            data = np.load(split_dir / f"traj_{i:05d}.npz")
            traj = {
                "observations": data["observations"],
                "actions": data["actions"],
                "rewards": data["rewards"],
                "return": float(data["ret"]),
                "length": int(data["length"]),
            }
            if "goals" in data.files:
                traj["goals"] = data["goals"]
            if "qpos_qvel_states" in data.files:
                traj["qpos_qvel_states"] = data["qpos_qvel_states"]
            trajs.append(traj)
        splits[split_name] = trajs

    return splits, stats
