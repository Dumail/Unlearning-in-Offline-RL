"""
uv run python scripts/run_layer_selection_cv.py
"""

import json
from pathlib import Path

import numpy as np


def load_selective_data(env_full: str) -> dict:
    """Load all selective results for one environment.

    Returns a {(target, steps, seed): gap} dict.
    """
    results = {}
    results_dir = Path(f"results/selective/{env_full}")
    if not results_dir.exists():
        return results

    for fpath in results_dir.glob("selective_*.json"):
        with open(fpath) as f:
            data = json.load(f)

        seed = data.get("seed", 0)
        steps = data.get("ascent_steps", 0)

        for target, r in data.get("results", {}).items():
            if "forget_auc" in r:
                gap = abs(r["forget_auc"] - 0.5)
                results[(target, steps, seed)] = {
                    "gap": gap,
                    "auc": r["forget_auc"],
                    "d4rl": r.get("d4rl_score", float("nan")),
                }

    return results


def run_cv(env_short: str, env_full: str, steps_list: list[int], seeds: list[int]):
    """Run leave-one-seed-out CV for one environment."""
    data = load_selective_data(env_full)
    if not data:
        print(f"  [skip] {env_short}: no data")
        return None

    # Collect all targets (excluding all and random)
    targets = set()
    for target, steps, seed in data:
        if target.startswith("attn") and not target.startswith("random"):
            targets.add(target)
    targets = sorted(targets)

    if not targets:
        print(f"  [skip] {env_short}: no attention target")
        return None

    print(f"  Available targets: {targets}")
    print(f"  Available seeds: {seeds}")
    print(f"  Available steps: {steps_list}")

    cv_results = []

    for held_out_seed in seeds:
        train_seeds = [s for s in seeds if s != held_out_seed]

        # For each target, compute mean gap across training seeds
        target_scores = {}
        for target in targets:
            gaps = []
            for steps in steps_list:
                for s in train_seeds:
                    key = (target, steps, s)
                    if key in data:
                        gaps.append(data[key]["gap"])
            if gaps:
                target_scores[target] = np.mean(gaps)

        if not target_scores:
            continue

        # CV selection: target with lowest mean gap on training seeds
        cv_best = min(target_scores, key=target_scores.get)

        # Compute performance on held-out seed
        cv_gaps = []
        oracle_best_gap = float("inf")
        oracle_best_target = None

        for target in targets:
            held_out_gaps = []
            for steps in steps_list:
                key = (target, steps, held_out_seed)
                if key in data:
                    held_out_gaps.append(data[key]["gap"])

            if held_out_gaps:
                mean_gap = np.mean(held_out_gaps)
                if target == cv_best:
                    cv_gaps = held_out_gaps
                if mean_gap < oracle_best_gap:
                    oracle_best_gap = mean_gap
                    oracle_best_target = target

        # Uniform baseline（all target）
        all_gaps = []
        for steps in steps_list:
            key = ("all", steps, held_out_seed)
            if key in data:
                all_gaps.append(data[key]["gap"])

        cv_gap_mean = np.mean(cv_gaps) if cv_gaps else float("nan")
        all_gap_mean = np.mean(all_gaps) if all_gaps else float("nan")

        result = {
            "held_out_seed": held_out_seed,
            "cv_selected": cv_best,
            "oracle_selected": oracle_best_target,
            "cv_gap": float(cv_gap_mean),
            "oracle_gap": float(oracle_best_gap),
            "uniform_gap": float(all_gap_mean),
            "regret_vs_oracle": float(cv_gap_mean - oracle_best_gap),
            "improvement_vs_uniform": float(all_gap_mean - cv_gap_mean),
            "correct_selection": cv_best == oracle_best_target,
        }
        cv_results.append(result)

        print(
            f"  seed={held_out_seed}: CV→{cv_best} (gap={cv_gap_mean:.4f}), "
            f"Oracle→{oracle_best_target} (gap={oracle_best_gap:.4f}), "
            f"All (gap={all_gap_mean:.4f}), "
            f"regret={cv_gap_mean - oracle_best_gap:+.4f}, "
            f"{'✓' if result['correct_selection'] else '✗'}"
        )

    return cv_results


def main():
    envs = {
        "hopper": "hopper-medium-replay-v2",
        "halfcheetah": "halfcheetah-medium-replay-v2",
        "walker2d": "walker2d-medium-replay-v2",
    }
    seeds = [0, 1, 2]
    steps_list = [100, 250, 500]

    print("=" * 80)
    print("Leave-One-Seed-Out Layer Selection Cross-Validation")
    print("=" * 80)

    all_results = {}

    for env_short, env_full in envs.items():
        print(f"\n--- {env_short} ---")
        cv = run_cv(env_short, env_full, steps_list, seeds)
        if cv:
            all_results[env_short] = cv

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for env_short, cv_list in all_results.items():
        correct = sum(1 for r in cv_list if r["correct_selection"])
        total = len(cv_list)
        mean_regret = np.mean([r["regret_vs_oracle"] for r in cv_list])
        mean_improvement = np.mean([r["improvement_vs_uniform"] for r in cv_list])

        print(
            f"  {env_short}: correct {correct}/{total}, "
            f"mean regret vs oracle: {mean_regret:+.4f}, "
            f"mean improvement vs uniform: {mean_improvement:+.4f}"
        )

    # Save results
    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_selection_cv.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
