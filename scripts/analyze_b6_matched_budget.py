#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from statistics import mean
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "analysis"


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if value != value:
            return "nan"
        return f"{float(value):.{digits}f}"
    return str(value)


def _load_records(results_dir: Path) -> list[dict[str, Any]]:
    records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/b6_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("block") != "B6":
            continue

        n_params = _safe_int(payload.get("n_params"))
        train_steps = _safe_int(payload.get("train_steps"))
        context_length = _safe_int(payload.get("context_length"))
        matched_budget_status = str(
            payload.get("matched_budget_status", "legacy_missing_metadata")
        )
        has_metadata = all(
            value is not None for value in (n_params, train_steps, context_length)
        )

        record = {
            "env": str(payload.get("env", path.parent.name)),
            "model": str(payload.get("model", "unknown")),
            "seed": int(payload.get("seed", -1)),
            "d4rl_score": _safe_float(payload.get("d4rl_score")),
            "forget_auc": _safe_float(payload.get("forget_auc")),
            "forget_auc_ci_low": _safe_float(payload.get("forget_auc_ci_low")),
            "forget_auc_ci_high": _safe_float(payload.get("forget_auc_ci_high")),
            "retain_diag_auc": _safe_float(payload.get("retain_diag_auc")),
            "n_params": n_params,
            "train_steps": train_steps,
            "context_length": context_length,
            "matched_budget_status": matched_budget_status,
            "has_budget_metadata": has_metadata,
            "file": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path),
        }
        key = (record["env"], record["model"], record["seed"])
        existing = records_by_key.get(key)
        if existing is None or "_seed" in record["file"]:
            records_by_key[key] = record
    return sorted(
        records_by_key.values(),
        key=lambda item: (item["env"], item["model"], item["seed"]),
    )


def _bootstrap_mean_ci(
    values: list[float], n_bootstrap: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr.item())
        return value, value, value
    rng = np.random.RandomState(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size
    for idx in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boots[idx] = float(sample.mean())
    return (
        float(arr.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def _coalesce_shared_int(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return present[0]


def _summarize(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_run_rows = sorted(
        records, key=lambda item: (item["env"], item["seed"], item["model"])
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["env"], record["seed"])].append(record)

    env_seed_rows: list[dict[str, Any]] = []
    for (env, seed), items in sorted(grouped.items()):
        n_params_values = [
            item["n_params"] for item in items if item["n_params"] is not None
        ]
        steps_values = [
            item["train_steps"] for item in items if item["train_steps"] is not None
        ]
        context_values = [
            item["context_length"]
            for item in items
            if item["context_length"] is not None
        ]
        status_counts = Counter(str(item["matched_budget_status"]) for item in items)
        params_complete = len(n_params_values) == len(items) and len(items) > 0
        steps_complete = len(steps_values) == len(items) and len(items) > 0
        contexts_complete = len(context_values) == len(items) and len(items) > 0

        max_min_ratio = float("nan")
        if params_complete and min(n_params_values) > 0:
            max_min_ratio = max(n_params_values) / min(n_params_values)

        env_seed_rows.append(
            {
                "env": env,
                "seed": seed,
                "models": ",".join(sorted(str(item["model"]) for item in items)),
                "n_models": len(items),
                "params_complete": params_complete,
                "train_steps_complete": steps_complete,
                "context_length_complete": contexts_complete,
                "max_param_ratio": max_min_ratio,
                "shared_train_steps": steps_complete and len(set(steps_values)) == 1,
                "shared_context_length": contexts_complete
                and len(set(context_values)) == 1,
                "status_summary": ";".join(
                    f"{name}:{count}" for name, count in sorted(status_counts.items())
                ),
            }
        )

    grouped_by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_by_model[(record["env"], record["model"])].append(record)

    multiseed_rows: list[dict[str, Any]] = []
    for (env, model), items in sorted(grouped_by_model.items()):
        items = sorted(items, key=lambda item: item["seed"])
        d4rl_values = [float(item["d4rl_score"]) for item in items]
        forget_values = [float(item["forget_auc"]) for item in items]
        retain_values = [float(item["retain_diag_auc"]) for item in items]
        d4rl_mean, d4rl_ci_low, d4rl_ci_high = _bootstrap_mean_ci(d4rl_values)
        forget_mean, forget_ci_low, forget_ci_high = _bootstrap_mean_ci(forget_values)
        retain_mean, retain_ci_low, retain_ci_high = _bootstrap_mean_ci(retain_values)
        multiseed_rows.append(
            {
                "env": env,
                "model": model,
                "n_seeds": len(items),
                "seed_list": ",".join(str(int(item["seed"])) for item in items),
                "d4rl_score": d4rl_mean,
                "d4rl_score_seed_ci_low": d4rl_ci_low,
                "d4rl_score_seed_ci_high": d4rl_ci_high,
                "forget_auc": forget_mean,
                "forget_auc_ci_low": forget_ci_low,
                "forget_auc_ci_high": forget_ci_high,
                "retain_diag_auc": retain_mean,
                "retain_diag_auc_ci_low": retain_ci_low,
                "retain_diag_auc_ci_high": retain_ci_high,
                "n_params": _coalesce_shared_int([item["n_params"] for item in items]),
                "train_steps": _coalesce_shared_int(
                    [item["train_steps"] for item in items]
                ),
                "context_length": _coalesce_shared_int(
                    [item["context_length"] for item in items]
                ),
                "matched_budget_status": ";".join(
                    sorted({str(item["matched_budget_status"]) for item in items})
                ),
                "has_budget_metadata": all(
                    bool(item["has_budget_metadata"]) for item in items
                ),
                "source_files": ";".join(str(item["file"]) for item in items),
            }
        )

    return per_run_rows, env_seed_rows, multiseed_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_markdown(
    per_run_rows: list[dict[str, Any]],
    env_seed_rows: list[dict[str, Any]],
    multiseed_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# B6 Matched-Budget Summary",
        "",
        "This file is auto-generated by `scripts/analyze_b6_matched_budget.py` to summarize parameter counts, training steps, and matched-budget metadata coverage across B6 backbone comparisons.",
        "",
        "## Env-Seed Coverage",
        "",
        "| Environment | Seed | Models | Params Complete | Shared Train Steps | Shared Context Length | Max Param Ratio | Status Summary |",
        "|---|---:|---|---|---|---|---:|---|",
    ]

    for row in env_seed_rows:
        lines.append(
            "| {env} | {seed} | {models} | {params_complete} | {shared_train_steps} | {shared_context_length} | {max_param_ratio} | {status_summary} |".format(
                env=row["env"],
                seed=row["seed"],
                models=row["models"],
                params_complete=row["params_complete"],
                shared_train_steps=row["shared_train_steps"],
                shared_context_length=row["shared_context_length"],
                max_param_ratio=_format_float(row["max_param_ratio"], digits=3),
                status_summary=row["status_summary"],
            )
        )

    lines.extend(
        [
            "",
            "## Env-Model Multi-Seed Summary",
            "",
            "| Environment | Model | Seeds | D4RL Mean | Forget AUC Mean | 95% CI | Retain AUC Mean | Params | Train Steps | Context |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )

    for row in multiseed_rows:
        lines.append(
            "| {env} | {model} | {seed_list} | {d4rl} | {forget_auc} | [{ci_low}, {ci_high}] | {retain_auc} | {params} | {train_steps} | {context_length} |".format(
                env=row["env"],
                model=row["model"],
                seed_list=row["seed_list"],
                d4rl=_format_float(row["d4rl_score"], digits=2),
                forget_auc=_format_float(row["forget_auc"]),
                ci_low=_format_float(row["forget_auc_ci_low"]),
                ci_high=_format_float(row["forget_auc_ci_high"]),
                retain_auc=_format_float(row["retain_diag_auc"]),
                params=row["n_params"] if row["n_params"] is not None else "missing",
                train_steps=row["train_steps"]
                if row["train_steps"] is not None
                else "missing",
                context_length=row["context_length"]
                if row["context_length"] is not None
                else "missing",
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Detail",
            "",
            "| Environment | Seed | Model | D4RL | Forget AUC | Retain AUC | Params | Train Steps | Context | Status | File |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )

    for row in per_run_rows:
        lines.append(
            "| {env} | {seed} | {model} | {d4rl} | {forget_auc} | {retain_auc} | {params} | {train_steps} | {context_length} | {status} | `{file}` |".format(
                env=row["env"],
                seed=row["seed"],
                model=row["model"],
                d4rl=_format_float(row["d4rl_score"], digits=2),
                forget_auc=_format_float(row["forget_auc"]),
                retain_auc=_format_float(row["retain_diag_auc"]),
                params=row["n_params"] if row["n_params"] is not None else "missing",
                train_steps=row["train_steps"]
                if row["train_steps"] is not None
                else "missing",
                context_length=row["context_length"]
                if row["context_length"] is not None
                else "missing",
                status=row["matched_budget_status"],
                file=row["file"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `legacy_missing_metadata`: Legacy B6 results containing only original contract fields, without parameter count or budget metadata.",
            "- `shared_train_steps`: Identical training steps recorded, but this does not imply parameter counts are strictly matched.",
            "- `max_param_ratio`: Ratio of max to min parameter count within the same environment and seed; only interpretable when all models have recorded `n_params`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize B6 matched-budget metadata")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Results root directory, default: results",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for analysis artifacts, default: results/analysis",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    per_run_rows, env_seed_rows, multiseed_rows = _summarize(_load_records(results_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "per_run": per_run_rows,
        "env_seed_summary": env_seed_rows,
        "env_model_multiseed": multiseed_rows,
    }

    json_path = output_dir / "b6_matched_budget.json"
    csv_path = output_dir / "b6_matched_budget.csv"
    multiseed_csv_path = output_dir / "b6_matched_budget_multiseed.csv"
    md_path = output_dir / "b6_matched_budget.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        csv_path,
        per_run_rows,
        [
            "env",
            "model",
            "seed",
            "d4rl_score",
            "forget_auc",
            "forget_auc_ci_low",
            "forget_auc_ci_high",
            "retain_diag_auc",
            "n_params",
            "train_steps",
            "context_length",
            "matched_budget_status",
            "has_budget_metadata",
            "file",
        ],
    )
    _write_csv(
        multiseed_csv_path,
        multiseed_rows,
        [
            "env",
            "model",
            "n_seeds",
            "seed_list",
            "d4rl_score",
            "d4rl_score_seed_ci_low",
            "d4rl_score_seed_ci_high",
            "forget_auc",
            "forget_auc_ci_low",
            "forget_auc_ci_high",
            "retain_diag_auc",
            "retain_diag_auc_ci_low",
            "retain_diag_auc_ci_high",
            "n_params",
            "train_steps",
            "context_length",
            "matched_budget_status",
            "has_budget_metadata",
            "source_files",
        ],
    )
    md_path.write_text(
        _build_markdown(per_run_rows, env_seed_rows, multiseed_rows),
        encoding="utf-8",
    )

    print(f"Loaded B6 rows: {len(per_run_rows)}")
    print(f"Summarized env-seed groups: {len(env_seed_rows)}")
    print(f"Summarized env-model groups: {len(multiseed_rows)}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Multi-seed CSV: {multiseed_csv_path}")
    print(f"Saved Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
