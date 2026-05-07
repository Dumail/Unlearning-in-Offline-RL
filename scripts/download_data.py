import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

hydra = importlib.import_module("hydra")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    compute_dataset_stats,
    create_forget_retain,
    download_dataset,
    extract_trajectories,
    filter_trajectories_by_return,
    format_forget_ratio,
    parse_forget_ratio_cli_arg,
    resolve_ratio_artifact_dir,
    save_splits,
    split_trajectories,
)
from src.antmaze_utils import resolve_antmaze_reward_mode


DEFAULT_FORGET_RATIO = 0.10


def _parse_forget_ratio_cli(default: float = DEFAULT_FORGET_RATIO) -> float:
    return parse_forget_ratio_cli_arg(sys.argv[1:], default=default)


CLI_FORGET_RATIO = _parse_forget_ratio_cli()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Any | None = None) -> None:
    if cfg is None:
        raise ValueError("Hydra cfg is required")
    data_dir = Path(cfg.data_dir)
    env_name = cfg.env.env_name
    forget_ratio = float(getattr(cfg, "forget_ratio", CLI_FORGET_RATIO))
    ratio_tag = format_forget_ratio(forget_ratio)
    trajectory_filter = str(getattr(cfg, "trajectory_filter", "all")).strip().lower()
    boundary_mode = (
        str(getattr(cfg, "trajectory_boundary_mode", "terminals_or_timeouts"))
        .strip()
        .lower()
    )
    antmaze_reward_mode = resolve_antmaze_reward_mode(
        str(env_name),
        getattr(cfg, "antmaze_reward_mode", None),
    )

    print(f"=== Data Pipeline: {env_name} ===")
    print(f"forget_ratio={ratio_tag}")
    print(f"trajectory_filter={trajectory_filter}")
    print(f"trajectory_boundary_mode={boundary_mode}")
    print(f"antmaze_reward_mode={antmaze_reward_mode}")

    # Step 1: Download (or locate) HDF5
    hdf5_path = download_dataset(
        env_name,
        str(data_dir),
        dataset_url=str(cfg.env.dataset_url),
    )
    print(f"HDF5 path: {hdf5_path}")

    # Step 2: Extract trajectories
    trajectories = extract_trajectories(
        hdf5_path,
        boundary_mode=boundary_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    if trajectory_filter == "positive_return":
        trajectories = filter_trajectories_by_return(
            trajectories, min_return_exclusive=0.0
        )
    elif trajectory_filter != "all":
        raise ValueError(
            "trajectory_filter only supports 'all' or 'positive_return', "
            f"received: {trajectory_filter}"
        )
    print(f"Total trajectories: {len(trajectories)}")
    returns = [t["return"] for t in trajectories]
    lengths = [t["length"] for t in trajectories]
    print(
        f"Return: mean={sum(returns) / len(returns):.1f}, "
        f"min={min(returns):.1f}, max={max(returns):.1f}"
    )
    print(
        f"Length: mean={sum(lengths) / len(lengths):.1f}, "
        f"min={min(lengths)}, max={max(lengths)}"
    )

    train, cal, test = split_trajectories(trajectories)
    forget, retain = create_forget_retain(
        train, forget_ratio=forget_ratio, seed=cfg.seed
    )

    stats = compute_dataset_stats(train)
    print(f"State mean shape: {stats['state_mean'].shape}")
    print(
        f"Train-only return stats: mean={stats['return_mean']:.1f}, "
        f"std={stats['return_std']:.1f}"
    )

    # Verify ratios
    total = len(trajectories)
    print(f"\n=== Split Verification ===")
    print(f"Total: {total}")
    print(f"Train: {len(train)} ({len(train) / total * 100:.1f}%)")
    print(f"  Forget: {len(forget)} ({len(forget) / len(train) * 100:.1f}% of train)")
    print(f"  Retain: {len(retain)} ({len(retain) / len(train) * 100:.1f}% of train)")
    print(f"Cal: {len(cal)} ({len(cal) / total * 100:.1f}%)")
    print(f"Test: {len(test)} ({len(test) / total * 100:.1f}%)")

    # Step 5: Save
    base_save_dir = data_dir / env_name
    if trajectory_filter != "all":
        base_save_dir = base_save_dir / trajectory_filter
    if boundary_mode != "terminals_or_timeouts":
        base_save_dir = base_save_dir / boundary_mode
    save_dir = resolve_ratio_artifact_dir(base_save_dir, forget_ratio=forget_ratio)
    splits = {
        "train": train,
        "calibration": cal,
        "test": test,
        "forget": forget,
        "retain": retain,
    }
    save_splits(
        splits,
        stats,
        save_dir,
        metadata={
            "forget_ratio": np.array(forget_ratio),
            "ratio_tag": np.array(ratio_tag),
            "seed": np.array(cfg.seed),
            "trajectory_filter": np.array(trajectory_filter),
            "trajectory_boundary_mode": np.array(boundary_mode),
            "antmaze_reward_mode": np.array(antmaze_reward_mode),
        },
    )
    print(f"\nAll data saved to {save_dir}")


if __name__ == "__main__":
    main()
