#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.experiment_contract import (
    canonical_b4_selection_specs,
    canonical_b4t_selection_specs,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_AUDIT_PATH = DEFAULT_RESULTS_ROOT / "analysis" / "benchmark_audit.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "analysis"

ENV_ORDER = {
    "halfcheetah-medium-replay-v2": 0,
    "hopper-medium-replay-v2": 1,
    "walker2d-medium-replay-v2": 2,
    "halfcheetah-medium-v2": 3,
    "hopper-medium-v2": 4,
    "walker2d-medium-v2": 5,
    "halfcheetah-medium-expert-v2": 6,
    "hopper-medium-expert-v2": 7,
    "walker2d-medium-expert-v2": 8,
    "antmaze-umaze-v2": 9,
    "antmaze-umaze-diverse-v2": 10,
    "antmaze-medium-diverse-v2": 11,
}

BLOCK_ORDER = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B4T": 4}

METHOD_LABELS = {
    "B1": "Base DT (B1)",
    "B2": "Gold Standard (B2)",
    "B3": "Naive FT (B3)",
    "B4": "GA+Refit (B4)",
    "B4T": "TrajDeleter (B4T)",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_name(name: str) -> int | None:
    if name == "tmi_eval_dt_final.json":
        return 0
    match = re.search(r"seed(\d+)", name)
    if match:
        return int(match.group(1))
    return None


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


def _candidate_files(
    env_name: str, block_name: str, files: list[str]
) -> tuple[list[str], str]:
    if block_name == "B1":
        selected = [
            name
            for name in files
            if name == "tmi_eval_dt_final.json"
            or re.fullmatch(r"tmi_eval_dt_seed\d+\.json", name)
        ]
        return sorted(
            selected,
            key=lambda name: (
                _seed_from_name(name) is None,
                _seed_from_name(name) or 0,
            ),
        ), "base_dt"
    if block_name == "B2":
        selected = [
            name for name in files if re.fullmatch(r"gold_standard_seed\d+\.json", name)
        ]
        return sorted(
            selected, key=lambda name: _seed_from_name(name) or 0
        ), "gold_standard"
    if block_name == "B3":
        selected = [
            name for name in files if re.fullmatch(r"naive_ft_seed\d+\.json", name)
        ]
        return sorted(selected, key=lambda name: _seed_from_name(name) or 0), "naive_ft"
    if block_name == "B4":
        for family, pattern in canonical_b4_selection_specs(env_name):
            selected = [name for name in files if pattern.fullmatch(name)]
            if selected:
                return sorted(
                    selected, key=lambda name: _seed_from_name(name) or 0
                ), family
        selected = [
            name for name in files if re.fullmatch(r"ga_refit_.*seed\d+\.json", name)
        ]
        return sorted(
            selected, key=lambda name: _seed_from_name(name) or 0
        ), "ga_refit_fallback"
    if block_name == "B4T":
        for family, pattern in canonical_b4t_selection_specs(env_name):
            selected = [name for name in files if pattern.fullmatch(name)]
            if selected:
                return sorted(
                    selected, key=lambda name: _seed_from_name(name) or 0
                ), family
        selected = [
            name for name in files if re.fullmatch(r"trajdeleter_.*seed\d+\.json", name)
        ]
        return sorted(
            selected, key=lambda name: _seed_from_name(name) or 0
        ), "trajdeleter_fallback"
    return [], "unknown"


def _coerce_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _gate_pass(payload: dict[str, Any]) -> bool:
    if payload.get("gold_standard_valid") is not None:
        return _coerce_bool(payload.get("gold_standard_valid"))
    if payload.get("retain_nll_shift_pass") is not None:
        return _coerce_bool(payload.get("retain_nll_shift_pass"))
    return False


def _summarize_block(
    env_name: str,
    block_name: str,
    block_entry: dict[str, Any],
    results_root: Path,
) -> dict[str, Any] | None:
    files = list(block_entry.get("files", []))
    paths = list(block_entry.get("paths", []))
    path_by_file = {
        str(Path(path).name): Path(ROOT / path)
        if not Path(path).is_absolute()
        else Path(path)
        for path in paths
    }
    selected_files, selection = _candidate_files(env_name, block_name, files)
    if not selected_files:
        return None

    records: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    env_dir = results_root / env_name
    for file_name in selected_files:
        payload_path = path_by_file.get(file_name, env_dir / file_name)
        if not payload_path.exists():
            continue
        payload = _load_json(payload_path)
        seed = int(payload.get("seed", _seed_from_name(file_name) or 0))
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        records.append(
            {
                "seed": seed,
                "file": file_name,
                "d4rl_score": float(payload.get("d4rl_score", float("nan"))),
                "forget_auc": float(payload.get("forget_auc", float("nan"))),
                "retain_diag_auc": float(payload.get("retain_diag_auc", float("nan"))),
                "gate_pass": _gate_pass(payload),
            }
        )

    if not records:
        return None

    records.sort(key=lambda item: item["seed"])
    d4rl_values = [record["d4rl_score"] for record in records]
    forget_values = [record["forget_auc"] for record in records]
    retain_values = [record["retain_diag_auc"] for record in records]
    d4rl_mean, d4rl_ci_low, d4rl_ci_high = _bootstrap_mean_ci(d4rl_values)
    forget_mean, forget_ci_low, forget_ci_high = _bootstrap_mean_ci(forget_values)
    retain_mean, retain_ci_low, retain_ci_high = _bootstrap_mean_ci(retain_values)
    valid_passes = sum(1 for record in records if record["gate_pass"])

    return {
        "env": env_name,
        "block": block_name,
        "method_label": METHOD_LABELS[block_name],
        "selection": selection,
        "n_seeds": len(records),
        "seeds": [record["seed"] for record in records],
        "source_files": [record["file"] for record in records],
        "d4rl_score_mean": d4rl_mean,
        "d4rl_score_seed_ci_low": d4rl_ci_low,
        "d4rl_score_seed_ci_high": d4rl_ci_high,
        "forget_auc_mean": forget_mean,
        "forget_auc_seed_ci_low": forget_ci_low,
        "forget_auc_seed_ci_high": forget_ci_high,
        "retain_diag_auc_mean": retain_mean,
        "retain_diag_auc_seed_ci_low": retain_ci_low,
        "retain_diag_auc_seed_ci_high": retain_ci_high,
        "gate_passes": valid_passes,
        "gate_pass_rate": valid_passes / len(records),
        "gold_standard_valid_passes": valid_passes,
        "gold_standard_valid_rate": valid_passes / len(records),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "env",
        "block",
        "method_label",
        "selection",
        "n_seeds",
        "seeds",
        "source_files",
        "d4rl_score_mean",
        "d4rl_score_seed_ci_low",
        "d4rl_score_seed_ci_high",
        "forget_auc_mean",
        "forget_auc_seed_ci_low",
        "forget_auc_seed_ci_high",
        "retain_diag_auc_mean",
        "retain_diag_auc_seed_ci_low",
        "retain_diag_auc_seed_ci_high",
        "gate_passes",
        "gate_pass_rate",
        "gold_standard_valid_passes",
        "gold_standard_valid_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            csv_row["source_files"] = ",".join(row["source_files"])
            writer.writerow(csv_row)


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# B1-B4/B4T Multi-seed Summary",
        "",
        "| Environment | Block | Method | Seeds | D4RL Mean | Forget AUC Mean | 95% CI | Retain AUC Mean | Gate Pass | Selection |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {env} | {block} | {method} | {n_seeds} | {d4rl:.2f} | {forget:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | {retain:.3f} | {passes}/{n_seeds} | {selection} |".format(
                env=row["env"],
                block=row["block"],
                method=row["method_label"],
                n_seeds=row["n_seeds"],
                d4rl=row["d4rl_score_mean"],
                forget=row["forget_auc_mean"],
                ci_low=row["forget_auc_seed_ci_low"],
                ci_high=row["forget_auc_seed_ci_high"],
                retain=row["retain_diag_auc_mean"],
                passes=row["gate_passes"],
                selection=row["selection"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize B1-B4/B4T multi-seed results"
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Results root directory, default: results",
    )
    parser.add_argument(
        "--audit-path",
        default=str(DEFAULT_AUDIT_PATH),
        help="Path to benchmark audit JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for analysis results, default: results/analysis",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    audit_path = Path(args.audit_path)
    output_dir = Path(args.output_dir)

    audit = _load_json(audit_path)
    environments = sorted(
        audit.get("environments", []), key=lambda item: ENV_ORDER.get(item["env"], 999)
    )

    rows: list[dict[str, Any]] = []
    for env_entry in environments:
        env_name = env_entry["env"]
        for block in env_entry.get("blocks", []):
            block_name = block.get("block")
            if block_name not in BLOCK_ORDER:
                continue
            row = _summarize_block(
                env_name=env_name,
                block_name=block_name,
                block_entry=block,
                results_root=results_root,
            )
            if row is not None:
                rows.append(row)

    rows.sort(
        key=lambda item: (ENV_ORDER.get(item["env"], 999), BLOCK_ORDER[item["block"]])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "b1_b4_multiseed_summary.json"
    csv_path = output_dir / "b1_b4_multiseed_summary.csv"
    md_path = output_dir / "b1_b4_multiseed_summary.md"

    json_path.write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    md_path.write_text(_build_markdown(rows), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
