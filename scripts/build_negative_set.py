"""Build matched non-member set from test trajectories."""

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

hydra = importlib.import_module("hydra")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import load_splits
from src.data_pipeline import (
    format_forget_ratio,
    load_split_metadata,
    parse_forget_ratio_cli_arg,
    require_forget_ratio_metadata,
    resolve_matched_negative_path,
    resolve_ratio_artifact_dir,
    resolve_split_dir,
)
from src.antmaze_utils import (
    compute_augmented_state_stats,
    resolve_antmaze_offline_state_mode,
)
from src.matching import (
    build_matched_negative_set,
    extract_trajectory_features,
    tune_matching_threshold,
)


DEFAULT_FORGET_RATIO = 0.10
DEFAULT_MATCHING_VARIANT = "basic"


def _parse_forget_ratio_cli(default: float = DEFAULT_FORGET_RATIO) -> float:
    return parse_forget_ratio_cli_arg(sys.argv[1:], default=default)


def resolve_matched_negative_save_path(data_dir: Path, forget_ratio: float) -> Path:
    return resolve_matched_negative_path(
        data_dir,
        forget_ratio,
        matching_variant=DEFAULT_MATCHING_VARIANT,
    )


CLI_FORGET_RATIO = _parse_forget_ratio_cli()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Any | None = None) -> None:
    if cfg is None:
        raise ValueError("Hydra cfg is required")
    base_data_dir = resolve_split_dir(cfg.data_dir, cfg.env.env_name)
    cli_forget_ratio = float(getattr(cfg, "forget_ratio", CLI_FORGET_RATIO))
    matching_variant = (
        str(getattr(cfg, "matching_variant", DEFAULT_MATCHING_VARIANT)).strip().lower()
    )
    antmaze_offline_state_mode = resolve_antmaze_offline_state_mode(
        str(cfg.env.env_name),
        getattr(cfg, "antmaze_offline_state_mode", None),
    )
    data_dir = resolve_ratio_artifact_dir(base_data_dir, forget_ratio=cli_forget_ratio)
    ratio_tag = format_forget_ratio(cli_forget_ratio)

    metadata_ratio = require_forget_ratio_metadata(data_dir)
    if not np.isclose(metadata_ratio, cli_forget_ratio):
        raise ValueError(
            "forget_ratio is inconsistent with data artifacts: "
            f"cli={ratio_tag}, metadata={format_forget_ratio(metadata_ratio)}"
        )

    splits, stats = load_splits(data_dir)
    metadata = load_split_metadata(data_dir)
    state_mean, state_std = compute_augmented_state_stats(
        splits["train"],
        goal_mode="none",
        offline_state_mode=antmaze_offline_state_mode,
    )
    feature_stats = dict(stats)
    feature_stats["state_mean"] = state_mean
    feature_stats["state_std"] = state_std

    print(f"=== Building Negative Set: {cfg.env.env_name} ===\n")
    print(f"forget_ratio={ratio_tag}")
    print(f"matching_variant={matching_variant}")

    # Extract features
    profile = "stronger" if matching_variant == "stronger" else "basic"
    train_feats = extract_trajectory_features(
        splits["train"],
        feature_stats,
        profile=profile,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )
    cal_feats = extract_trajectory_features(
        splits["calibration"],
        feature_stats,
        profile=profile,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )
    forget_feats = extract_trajectory_features(
        splits["forget"],
        feature_stats,
        profile=profile,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )
    test_feats = extract_trajectory_features(
        splits["test"],
        feature_stats,
        profile=profile,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
    )

    print(f"Feature dimensions: {train_feats.shape[1]}")
    print(
        f"Train: {len(train_feats)}, Cal: {len(cal_feats)}, "
        f"Forget: {len(forget_feats)}, Test: {len(test_feats)}\n"
    )

    # Tune threshold (use full_coverage when test pool is larger than forget set)
    calibration_source = cal_feats
    if len(calibration_source) == 0:
        calibration_source = train_feats
        print(
            "[WARN] calibration split is empty; falling back to train features "
            "for threshold tuning"
        )

    use_full_coverage = len(test_feats) > len(forget_feats)
    threshold = tune_matching_threshold(
        train_feats,
        calibration_source,
        forget_ratio=cli_forget_ratio,
        seed=cfg.seed,
        full_coverage=use_full_coverage,
    )

    # Build matched non-member set
    print()
    matched_pairs, quality = build_matched_negative_set(
        forget_feats,
        test_feats,
        splits["test"],
        threshold,
    )

    # Print quality report
    print(f"\n=== Quality Report ===")
    print(f"Forget set size: {quality['n_forget']}")
    print(f"Matched: {quality['n_matched']} ({quality['fraction_matched'] * 100:.1f}%)")
    print(f"Threshold: {quality['threshold']:.4f}")
    print(
        "Eligible candidates per forget: "
        f"mean={quality['eligible_count_mean']:.2f}, "
        f"min={quality['eligible_count_min']}, max={quality['eligible_count_max']}"
    )
    print(f"Zero-eligible forget trajectories: {quality['n_with_zero_eligible']}")
    print(
        "Unmatched breakdown: "
        f"threshold={quality['unmatched_due_to_threshold']}, "
        f"assignment={quality['unmatched_due_to_assignment']}"
    )

    if quality["n_matched"] > 0:
        print(
            f"Distance: mean={quality['distance_mean']:.4f}, "
            f"std={quality['distance_std']:.4f}, max={quality['distance_max']:.4f}"
        )
        print(
            f"Nearest distance: mean={quality['nearest_distance_mean']:.4f}, "
            f"std={quality['nearest_distance_std']:.4f}, "
            f"max={quality['nearest_distance_max']:.4f}"
        )
        print(f"Return delta (z-score): {quality['return_delta_mean']:.4f}")
        print(f"Length delta (z-score): {quality['length_delta_mean']:.4f}")

    # Save matched pairs
    save_path = resolve_matched_negative_path(
        data_dir,
        cli_forget_ratio,
        matching_variant=matching_variant,
    )
    forget_indices = [p[0] for p in matched_pairs]
    test_indices = [p[1] for p in matched_pairs]
    payload: dict[str, Any] = {
        "forget_indices": np.array(forget_indices, dtype=np.int64),
        "test_indices": np.array(test_indices, dtype=np.int64),
        "forget_ratio": np.array(cli_forget_ratio),
        "ratio_tag": np.array(ratio_tag),
        "matching_variant": np.array(matching_variant),
        "threshold": np.array(threshold),
        "split_metadata_ratio": np.array(metadata.get("forget_ratio", np.nan)),
    }
    for key, value in quality.items():
        payload[f"quality_{key}"] = np.array(value)
    np.savez(save_path, **payload)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
