from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTIVE_DIR = ROOT / "results" / "selective"
DEFAULT_CV_PATH = ROOT / "results" / "analysis" / "layer_selection_cv.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "analysis" / "selective_utility_budget"
DEFAULT_RESULTS_ROOT = ROOT / "results"
SEED_PATTERN = re.compile(r"(?:^|_)seed(\d+)(?:_|\.|$)", re.IGNORECASE)
STEPS_PATTERN = re.compile(r"(?:^|_)steps(\d+)(?:_|\.|$)", re.IGNORECASE)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _extract_seed(payload: dict[str, Any], path: Path) -> int:
    raw_seed = payload.get("seed")
    if raw_seed is not None:
        return int(raw_seed)
    matched = SEED_PATTERN.search(path.name)
    if matched:
        return int(matched.group(1))
    raise KeyError("seed")


def _extract_ascent_steps(payload: dict[str, Any], path: Path) -> int:
    raw_steps = payload.get("ascent_steps")
    if raw_steps is not None:
        return int(raw_steps)
    matched = STEPS_PATTERN.search(path.name)
    if matched:
        return int(matched.group(1))
    raise KeyError("ascent_steps")


def _load_rows(selective_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(selective_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if "env" not in payload:
            continue
        results = payload.get("results", {})
        if not isinstance(results, dict) or len(results) != 1:
            continue

        target, metrics = next(iter(results.items()))
        if not isinstance(metrics, dict):
            continue
        metrics_dict = {str(key): value for key, value in metrics.items()}

        forget_auc = _safe_float(metrics_dict.get("forget_auc"))
        rows.append(
            {
                "env": str(payload["env"]),
                "seed": _extract_seed(payload, path),
                "ascent_steps": _extract_ascent_steps(payload, path),
                "target": str(target),
                "forget_auc": forget_auc,
                "forget_gap": abs(forget_auc - 0.5),
                "d4rl_score": _safe_float(metrics_dict.get("d4rl_score")),
                "retain_diag_auc": _safe_float(metrics_dict.get("retain_diag_auc")),
                "gold_standard_valid": bool(
                    metrics_dict.get("gold_standard_valid", False)
                ),
                "retain_shift_pass": bool(
                    metrics_dict.get("retain_nll_shift_pass", False)
                ),
                "file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def _load_cv_targets(cv_path: Path) -> dict[tuple[str, int], str]:
    payload = json.loads(cv_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    env_map = {
        "hopper": "hopper-medium-replay-v2",
        "halfcheetah": "halfcheetah-medium-replay-v2",
        "walker2d": "walker2d-medium-replay-v2",
    }
    targets: dict[tuple[str, int], str] = {}
    for env_short, items in payload.items():
        env_name = env_map.get(env_short)
        if env_name is None:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            targets[(env_name, int(item["held_out_seed"]))] = str(item["cv_selected"])
    return targets


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator != denominator or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _load_reference_scores(
    results_root: Path, rows: list[dict]
) -> dict[tuple[str, int], dict[str, float]]:
    references: dict[tuple[str, int], dict[str, float]] = {}
    env_seeds = sorted({(str(row["env"]), int(row["seed"])) for row in rows})
    for env_name, seed in env_seeds:
        env_dir = results_root / env_name
        base_candidates = [env_dir / f"tmi_eval_dt_seed{seed}.json"]
        if seed == 0:
            base_candidates.append(env_dir / "tmi_eval_dt_final.json")
        gold_candidates = [env_dir / f"gold_standard_seed{seed}.json"]

        base_score = float("nan")
        gold_score = float("nan")

        for path in base_candidates:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "d4rl_score" in payload:
                    base_score = float(payload["d4rl_score"])
                    break

        for path in gold_candidates:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "d4rl_score" in payload:
                    gold_score = float(payload["d4rl_score"])
                    break

        references[(env_name, seed)] = {
            "base_d4rl": base_score,
            "gold_d4rl": gold_score,
        }
    return references


def build_reports(
    rows: list[dict],
    cv_targets: dict[tuple[str, int], str],
    budgets: list[float],
    references: dict[tuple[str, int], dict[str, float]],
    min_score_over_gold: float,
    min_score_over_base: float,
) -> tuple[list[dict], list[dict]]:
    by_key = {
        (row["env"], row["seed"], row["ascent_steps"], row["target"]): row
        for row in rows
    }

    comparison_rows: list[dict] = []
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)

    env_seed_steps = sorted(
        {(row["env"], row["seed"], row["ascent_steps"]) for row in rows}
    )
    for env_name, seed, ascent_steps in env_seed_steps:
        cv_target = cv_targets.get((env_name, seed))
        if cv_target is None:
            continue

        uniform = by_key.get((env_name, seed, ascent_steps, "all"))
        selected = by_key.get((env_name, seed, ascent_steps, cv_target))
        if uniform is None or selected is None:
            continue

        reference = references.get((env_name, seed), {})
        gold_d4rl = float(reference.get("gold_d4rl", float("nan")))
        base_d4rl = float(reference.get("base_d4rl", float("nan")))

        utility_delta = float(selected["d4rl_score"] - uniform["d4rl_score"])
        utility_loss = max(0.0, -utility_delta)
        gap_improvement = float(uniform["forget_gap"] - selected["forget_gap"])
        selected_score_over_gold = _safe_ratio(float(selected["d4rl_score"]), gold_d4rl)
        selected_score_over_base = _safe_ratio(float(selected["d4rl_score"]), base_d4rl)
        uniform_score_over_gold = _safe_ratio(float(uniform["d4rl_score"]), gold_d4rl)
        uniform_score_over_base = _safe_ratio(float(uniform["d4rl_score"]), base_d4rl)
        selected_absolute_pass = (
            selected_score_over_gold >= min_score_over_gold
            and selected_score_over_base >= min_score_over_base
        )
        uniform_absolute_pass = (
            uniform_score_over_gold >= min_score_over_gold
            and uniform_score_over_base >= min_score_over_base
        )

        for budget in budgets:
            row = {
                "env": env_name,
                "seed": seed,
                "ascent_steps": ascent_steps,
                "cv_target": cv_target,
                "budget": float(budget),
                "gold_d4rl": gold_d4rl,
                "base_d4rl": base_d4rl,
                "selected_gap": float(selected["forget_gap"]),
                "uniform_gap": float(uniform["forget_gap"]),
                "gap_improvement": gap_improvement,
                "selected_d4rl": float(selected["d4rl_score"]),
                "uniform_d4rl": float(uniform["d4rl_score"]),
                "selected_score_over_gold": selected_score_over_gold,
                "selected_score_over_base": selected_score_over_base,
                "uniform_score_over_gold": uniform_score_over_gold,
                "uniform_score_over_base": uniform_score_over_base,
                "utility_delta": utility_delta,
                "utility_loss": utility_loss,
                "within_budget": utility_loss <= budget,
                "selected_absolute_pass": selected_absolute_pass,
                "uniform_absolute_pass": uniform_absolute_pass,
                "absolute_pass_within_budget": utility_loss <= budget
                and selected_absolute_pass,
                "privacy_better": gap_improvement > 0.0,
                "privacy_better_within_budget": utility_loss <= budget
                and gap_improvement > 0.0,
                "privacy_better_absolute_pass": gap_improvement > 0.0
                and selected_absolute_pass,
                "privacy_better_absolute_pass_within_budget": utility_loss <= budget
                and gap_improvement > 0.0
                and selected_absolute_pass,
            }
            comparison_rows.append(row)
            grouped[(env_name, float(budget))].append(row)

    summary_rows: list[dict] = []
    for (env_name, budget), items in sorted(grouped.items()):
        feasible = [item for item in items if item["within_budget"]]
        successful = [item for item in feasible if item["privacy_better"]]
        absolute_feasible = [
            item for item in feasible if item["selected_absolute_pass"]
        ]
        absolute_success = [
            item for item in absolute_feasible if item["privacy_better"]
        ]
        summary_rows.append(
            {
                "env": env_name,
                "budget": budget,
                "n_total": len(items),
                "n_feasible": len(feasible),
                "n_success": len(successful),
                "n_absolute_feasible": len(absolute_feasible),
                "n_absolute_success": len(absolute_success),
                "feasible_rate": len(feasible) / len(items) if items else float("nan"),
                "success_rate_total": len(successful) / len(items)
                if items
                else float("nan"),
                "success_rate_feasible": len(successful) / len(feasible)
                if feasible
                else float("nan"),
                "absolute_feasible_rate": len(absolute_feasible) / len(items)
                if items
                else float("nan"),
                "absolute_success_rate_total": len(absolute_success) / len(items)
                if items
                else float("nan"),
                "absolute_success_rate_feasible": len(absolute_success)
                / len(absolute_feasible)
                if absolute_feasible
                else float("nan"),
                "selected_gap_mean": _mean([item["selected_gap"] for item in feasible]),
                "uniform_gap_mean": _mean([item["uniform_gap"] for item in feasible]),
                "gap_improvement_mean": _mean(
                    [item["gap_improvement"] for item in feasible]
                ),
                "selected_d4rl_mean": _mean(
                    [item["selected_d4rl"] for item in feasible]
                ),
                "uniform_d4rl_mean": _mean([item["uniform_d4rl"] for item in feasible]),
                "utility_delta_mean": _mean(
                    [item["utility_delta"] for item in feasible]
                ),
                "utility_loss_mean": _mean([item["utility_loss"] for item in feasible]),
                "selected_score_over_gold_mean": _mean(
                    [item["selected_score_over_gold"] for item in feasible]
                ),
                "selected_score_over_base_mean": _mean(
                    [item["selected_score_over_base"] for item in feasible]
                ),
                "uniform_score_over_gold_mean": _mean(
                    [item["uniform_score_over_gold"] for item in feasible]
                ),
                "uniform_score_over_base_mean": _mean(
                    [item["uniform_score_over_base"] for item in feasible]
                ),
                "cv_target_set": ", ".join(
                    sorted({item["cv_target"] for item in items})
                ),
            }
        )

    return comparison_rows, summary_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def _build_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "# Utility-Constrained Selective Comparison",
        "",
        "This report compares the cross-validated selective target against the uniform all-layer baseline at matched ascent steps.",
        "A row is feasible when the utility loss of the selective target, measured as the D4RL drop relative to the same-step uniform baseline, does not exceed the stated budget.",
        "Absolute utility pass additionally requires the selective target to satisfy the configured retained-performance ratios against both gold retrain and base DT.",
        "",
        "| Environment | Budget | Feasible / Total | Absolute Feasible / Total | Absolute Success / Total | Gap Improvement Mean | Score/Gold Mean | Score/Base Mean | CV Target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {env} | {budget} | {feasible}/{total} | {absolute_feasible}/{total} | {absolute_success}/{total} | {gap} | {score_gold} | {score_base} | {target} |".format(
                env=row["env"],
                budget=_fmt(row["budget"], digits=1),
                feasible=row["n_feasible"],
                total=row["n_total"],
                absolute_feasible=row["n_absolute_feasible"],
                absolute_success=row["n_absolute_success"],
                gap=_fmt(row["gap_improvement_mean"]),
                score_gold=_fmt(row["selected_score_over_gold_mean"]),
                score_base=_fmt(row["selected_score_over_base_mean"]),
                target=row["cv_target_set"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze utility-constrained selective comparison"
    )
    parser.add_argument(
        "--selective-dir",
        type=Path,
        default=DEFAULT_SELECTIVE_DIR,
        help="Selective results directory",
    )
    parser.add_argument(
        "--cv-path",
        type=Path,
        default=DEFAULT_CV_PATH,
        help="Layer-selection CV results path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Benchmark main results directory",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 2.0, 5.0],
        help="Allowed D4RL utility loss budget",
    )
    parser.add_argument(
        "--min-score-over-gold",
        type=float,
        default=1.0,
        help="absolute utility gate: selective score / gold score lower bound",
    )
    parser.add_argument(
        "--min-score-over-base",
        type=float,
        default=1.0,
        help="absolute utility gate: selective score / base score lower bound",
    )
    args = parser.parse_args()

    rows = _load_rows(args.selective_dir)
    cv_targets = _load_cv_targets(args.cv_path)
    references = _load_reference_scores(args.results_root, rows)
    comparison_rows, summary_rows = build_reports(
        rows,
        cv_targets,
        sorted(args.budgets),
        references,
        args.min_score_over_gold,
        args.min_score_over_base,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.output_dir / "selective_utility_budget_comparisons.csv",
        comparison_rows,
        [
            "env",
            "seed",
            "ascent_steps",
            "cv_target",
            "budget",
            "gold_d4rl",
            "base_d4rl",
            "selected_gap",
            "uniform_gap",
            "gap_improvement",
            "selected_d4rl",
            "uniform_d4rl",
            "selected_score_over_gold",
            "selected_score_over_base",
            "uniform_score_over_gold",
            "uniform_score_over_base",
            "utility_delta",
            "utility_loss",
            "within_budget",
            "selected_absolute_pass",
            "uniform_absolute_pass",
            "absolute_pass_within_budget",
            "privacy_better",
            "privacy_better_within_budget",
            "privacy_better_absolute_pass",
            "privacy_better_absolute_pass_within_budget",
        ],
    )
    _write_csv(
        args.output_dir / "selective_utility_budget_summary.csv",
        summary_rows,
        [
            "env",
            "budget",
            "n_total",
            "n_feasible",
            "n_success",
            "n_absolute_feasible",
            "n_absolute_success",
            "feasible_rate",
            "success_rate_total",
            "success_rate_feasible",
            "absolute_feasible_rate",
            "absolute_success_rate_total",
            "absolute_success_rate_feasible",
            "selected_gap_mean",
            "uniform_gap_mean",
            "gap_improvement_mean",
            "selected_d4rl_mean",
            "uniform_d4rl_mean",
            "utility_delta_mean",
            "utility_loss_mean",
            "selected_score_over_gold_mean",
            "selected_score_over_base_mean",
            "uniform_score_over_gold_mean",
            "uniform_score_over_base_mean",
            "cv_target_set",
        ],
    )
    (args.output_dir / "selective_utility_budget_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selective_utility_budget_summary.md").write_text(
        _build_markdown(summary_rows),
        encoding="utf-8",
    )

    print(f"Comparison rows: {len(comparison_rows)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Saved: {args.output_dir / 'selective_utility_budget_summary.md'}")


if __name__ == "__main__":
    main()
