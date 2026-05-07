#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "selective"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "analysis"
SEED_PATTERN = re.compile(r"(?:^|_)seed(\d+)(?:_|\.|$)", re.IGNORECASE)
STEPS_PATTERN = re.compile(r"(?:^|_)steps(\d+)(?:_|\.|$)", re.IGNORECASE)


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _gap(auc: float) -> float:
    return abs(auc - 0.5)


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


def _is_supported_target(target: str) -> bool:
    allowed_prefixes = ("all", "attn", "mlp")
    return target.startswith(allowed_prefixes) and not target.startswith("random_")


def _load_clean_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        results = payload.get("results", {})
        if not isinstance(results, dict) or len(results) != 1:
            continue

        target, metrics = next(iter(results.items()))
        if not _is_supported_target(str(target)):
            continue
        if not isinstance(metrics, dict):
            continue
        metrics_dict = {str(key): value for key, value in metrics.items()}

        forget_auc = _safe_float(metrics_dict.get("forget_auc"))
        d4rl_score = _safe_float(metrics_dict.get("d4rl_score"))
        retain_diag_auc = _safe_float(metrics_dict.get("retain_diag_auc"))
        record = {
            "env": str(payload.get("env", path.parent.name)),
            "seed": _extract_seed(payload, path),
            "ascent_steps": _extract_ascent_steps(payload, path),
            "target": str(target),
            "forget_auc": forget_auc,
            "forget_gap": _gap(forget_auc),
            "d4rl_score": d4rl_score,
            "retain_diag_auc": retain_diag_auc,
            "gold_standard_valid": bool(metrics_dict.get("gold_standard_valid", False)),
            "retain_nll_shift_pass": bool(
                metrics_dict.get("retain_nll_shift_pass", False)
            ),
            "file": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path),
        }
        records.append(record)
    return records


def _filter_envs(
    records: list[dict[str, Any]], selected_envs: set[str] | None
) -> list[dict[str, Any]]:
    if not selected_envs:
        return records
    return [record for record in records if str(record["env"]) in selected_envs]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarize(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["env"], record["ascent_steps"], record["target"])].append(
            record
        )

    baseline_by_env_step: dict[tuple[str, int], dict[str, Any]] = {}
    for (env, steps, target), items in grouped.items():
        if target != "all":
            continue
        baseline_by_env_step[(env, steps)] = {
            "forget_gap_mean": _mean([item["forget_gap"] for item in items]),
            "d4rl_mean": _mean([item["d4rl_score"] for item in items]),
        }

    summary_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    by_env_step: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (env, steps, target), items in sorted(grouped.items()):
        n_seeds = len(items)
        row = {
            "env": env,
            "ascent_steps": steps,
            "target": target,
            "n_seeds": n_seeds,
            "forget_auc_mean": _mean([item["forget_auc"] for item in items]),
            "forget_gap_mean": _mean([item["forget_gap"] for item in items]),
            "d4rl_mean": _mean([item["d4rl_score"] for item in items]),
            "retain_diag_auc_mean": _mean([item["retain_diag_auc"] for item in items]),
            "gold_standard_valid_rate": _mean(
                [1.0 if item["gold_standard_valid"] else 0.0 for item in items]
            ),
            "retain_shift_pass_rate": _mean(
                [1.0 if item["retain_nll_shift_pass"] else 0.0 for item in items]
            ),
        }

        baseline = baseline_by_env_step.get((env, steps))
        if baseline is not None:
            row["gap_delta_vs_all"] = (
                row["forget_gap_mean"] - baseline["forget_gap_mean"]
            )
            row["d4rl_delta_vs_all"] = row["d4rl_mean"] - baseline["d4rl_mean"]
        else:
            row["gap_delta_vs_all"] = float("nan")
            row["d4rl_delta_vs_all"] = float("nan")

        summary_rows.append(row)
        by_env_step[(env, steps)].append(row)

    for (env, steps), rows in sorted(by_env_step.items()):
        best = min(rows, key=lambda row: row["forget_gap_mean"])
        best_rows.append(
            {
                "env": env,
                "ascent_steps": steps,
                "best_target": best["target"],
                "forget_gap_mean": best["forget_gap_mean"],
                "forget_auc_mean": best["forget_auc_mean"],
                "d4rl_mean": best["d4rl_mean"],
            }
        )

    return summary_rows, best_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _build_markdown(
    summary_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "# Selective Result Summary",
        "",
        "## Aggregate By Step And Target",
        "",
        "| Environment | Steps | Target | Seeds | Gap Mean | AUC Mean | D4RL Mean | Gap Delta vs All | D4RL Delta vs All | GSV Rate | Shift Pass Rate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in sorted(
        summary_rows,
        key=lambda item: (item["env"], item["ascent_steps"], item["target"]),
    ):
        lines.append(
            "| {env} | {steps} | {target} | {n_seeds} | {gap} | {auc} | {d4rl} | {gap_delta} | {d4rl_delta} | {gsv} | {shift} |".format(
                env=row["env"],
                steps=row["ascent_steps"],
                target=row["target"],
                n_seeds=row["n_seeds"],
                gap=_format_float(row["forget_gap_mean"]),
                auc=_format_float(row["forget_auc_mean"]),
                d4rl=_format_float(row["d4rl_mean"], digits=2),
                gap_delta=_format_float(row["gap_delta_vs_all"]),
                d4rl_delta=_format_float(row["d4rl_delta_vs_all"], digits=2),
                gsv=_format_float(row["gold_standard_valid_rate"], digits=2),
                shift=_format_float(row["retain_shift_pass_rate"], digits=2),
            )
        )

    lines.extend(
        [
            "",
            "## Best Target By Step",
            "",
            "| Environment | Steps | Best Target | Gap Mean | AUC Mean | D4RL Mean |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(best_rows, key=lambda item: (item["env"], item["ascent_steps"])):
        lines.append(
            "| {env} | {steps} | {target} | {gap} | {auc} | {d4rl} |".format(
                env=row["env"],
                steps=row["ascent_steps"],
                target=row["best_target"],
                gap=_format_float(row["forget_gap_mean"]),
                auc=_format_float(row["forget_auc_mean"]),
                d4rl=_format_float(row["d4rl_mean"], digits=2),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize selective unlearning clean-baseline results"
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Selective results root directory, default: results/selective",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for analysis artifacts, default: results/analysis",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Only analyze the given environment name list; no filtering by default",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    selected_envs = set(args.envs) if args.envs else None
    records = _filter_envs(_load_clean_records(results_dir), selected_envs)
    summary_rows, best_rows = _summarize(records)

    summary_csv = output_dir / "selective_summary.csv"
    best_csv = output_dir / "selective_best_by_step.csv"
    summary_md = output_dir / "selective_summary.md"

    _write_csv(
        summary_csv,
        summary_rows,
        [
            "env",
            "ascent_steps",
            "target",
            "n_seeds",
            "forget_auc_mean",
            "forget_gap_mean",
            "d4rl_mean",
            "retain_diag_auc_mean",
            "gold_standard_valid_rate",
            "retain_shift_pass_rate",
            "gap_delta_vs_all",
            "d4rl_delta_vs_all",
        ],
    )
    _write_csv(
        best_csv,
        best_rows,
        [
            "env",
            "ascent_steps",
            "best_target",
            "forget_gap_mean",
            "forget_auc_mean",
            "d4rl_mean",
        ],
    )
    summary_md.write_text(_build_markdown(summary_rows, best_rows), encoding="utf-8")

    envs = sorted({row["env"] for row in summary_rows})
    print(f"Loaded clean selective rows: {len(records)}")
    print(f"Aggregated rows: {len(summary_rows)}")
    print(f"Environments: {envs}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved best-by-step CSV: {best_csv}")
    print(f"Saved summary Markdown: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
