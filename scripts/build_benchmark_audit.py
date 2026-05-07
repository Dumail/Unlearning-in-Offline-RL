#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_contract import (
    B7_REQUIRED_RATIOS,
    CANONICAL_ENV_SPECS,
    canonical_b4_selection_specs,
    canonical_b4t_selection_specs,
    expected_blocks_for_env,
)
from src.experiment_validator import (
    _read_records,
    related_results_dirs,
    validate_results_completeness,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "analysis"

TARGET_ENVS = tuple(spec.env_name for spec in CANONICAL_ENV_SPECS)

EXPECTED_BLOCKS = {
    env_name: expected_blocks_for_env(env_name) for env_name in TARGET_ENVS
}

PLACEHOLDER_B6 = {
    "dt": {
        "d4rl_score": 20.0,
        "forget_auc": 0.6,
        "forget_auc_ci_low": 0.55,
        "forget_auc_ci_high": 0.65,
        "retain_diag_auc": 0.52,
    },
    "mlp": {
        "d4rl_score": 20.01,
        "forget_auc": 0.61,
        "forget_auc_ci_low": 0.56,
        "forget_auc_ci_high": 0.66,
        "retain_diag_auc": 0.53,
    },
}

PLACEHOLDER_B7 = {
    "dt": {
        "d4rl_score": 20.0,
        "forget_auc": 0.6,
        "forget_auc_ci_low": 0.55,
        "forget_auc_ci_high": 0.65,
        "retain_diag_auc": 0.52,
    },
    "mlp": {
        "d4rl_score": 20.0,
        "forget_auc": 0.6,
        "forget_auc_ci_low": 0.55,
        "forget_auc_ci_high": 0.65,
        "retain_diag_auc": 0.52,
    },
}


def _read_json(path: Path) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _matches_signature(payload: dict[str, Any], signature: dict[str, Any]) -> bool:
    for key, expected in signature.items():
        if payload.get(key) != expected:
            return False
    return True


def _classify_b1(path: Path, payload: dict[str, Any]) -> str:
    method = payload.get("method")
    if path.name == "tmi_eval_dt_final.json" and method == "base":
        return "canonical_export"
    if path.name.startswith("tmi_eval_dt_seed") and method == "base":
        return "canonical_export"
    return "real_run"


def _classify_b6(payload: dict[str, Any]) -> str:
    model = str(payload.get("model", "")).lower()
    signature = PLACEHOLDER_B6.get(model)
    if signature is not None and _matches_signature(payload, signature):
        return "contract_placeholder"
    return "real_run"


def _classify_b7(payload: dict[str, Any]) -> str:
    model = str(payload.get("model", "")).lower()
    signature = PLACEHOLDER_B7.get(model)
    if signature is not None and _matches_signature(payload, signature):
        return "contract_placeholder"
    return "real_run"


def _classify_record(path: Path, payload: dict[str, Any]) -> str:
    block = payload.get("block")
    if block == "B1":
        return _classify_b1(path, payload)
    if block in {"B2", "B3", "B4", "B4T", "B5"}:
        return "real_run"
    if block == "B6":
        return _classify_b6(payload)
    if block == "B7":
        return _classify_b7(payload)
    return "unknown"


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _discover_records(env_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for record in _read_records(env_dir, errors):
        payload = record.payload
        block = record.block
        if not isinstance(block, str):
            continue
        if "block" not in payload:
            payload = dict(payload)
            payload["block"] = block
        rows.append(
            {
                "env": env_dir.name,
                "source_dir": _artifact_path(record.path.parent),
                "block": block,
                "file": record.path.name,
                "path": _artifact_path(record.path),
                "provenance": _classify_record(record.path, payload),
                "model": payload.get("model"),
                "seed": payload.get("seed"),
                "forget_ratio": payload.get("forget_ratio"),
                "method": payload.get("method"),
                "d4rl_score": payload.get("d4rl_score"),
                "forget_auc": payload.get("forget_auc"),
                "retain_diag_auc": payload.get("retain_diag_auc"),
                "gold_standard_valid": payload.get("gold_standard_valid"),
                "retain_nll_shift_pass": payload.get("retain_nll_shift_pass"),
            }
        )
    return rows


def _gate_status(block: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"

    if block not in {"B2", "B3", "B4", "B4T"}:
        return "not_applicable"

    saw_explicit_gate = False

    for row in rows:
        gold_valid = row.get("gold_standard_valid")
        retain_shift_pass = row.get("retain_nll_shift_pass")

        if isinstance(gold_valid, bool):
            saw_explicit_gate = True
            if not gold_valid:
                return "failed"

        if isinstance(retain_shift_pass, bool):
            saw_explicit_gate = True
            if not retain_shift_pass:
                return "failed"

    if saw_explicit_gate:
        return "passed"
    return "unknown"


def _select_canonical_b4_rows(
    env: str, block_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    env_key = str(env).strip().lower()
    if not env_key.startswith("antmaze-"):
        return block_rows, None

    file_to_row = {str(row["file"]): row for row in block_rows}
    files = list(file_to_row)
    for selection, pattern in canonical_b4_selection_specs(env_key):
        selected_files = [name for name in files if pattern.fullmatch(name)]
        if selected_files:
            return [file_to_row[name] for name in selected_files], selection
    return block_rows, None


def _select_antmaze_preferred_rows(
    env: str, block: str, block_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    env_key = str(env).strip().lower()
    if not env_key.startswith("antmaze-"):
        return block_rows, None

    analysis_base = f"results/analysis/antmaze_fixed_goal_base_multiseed/{env_key}/"
    analysis_four = f"results/analysis/antmaze_four_method_runs/{env_key}/"
    root_prefix = f"results/{env_key}/"

    def _in_prefix(row: dict[str, Any], prefix: str) -> bool:
        return str(row.get("path", "")).startswith(prefix)

    def _seed(row: dict[str, Any]) -> int:
        try:
            return int(row.get("seed", 0))
        except Exception:
            return 0

    if block == "B1":
        root_seed0 = [
            row
            for row in block_rows
            if _in_prefix(row, root_prefix)
            and str(row.get("file")) == "tmi_eval_dt_final.json"
        ]
        analysis_seed12 = [row for row in block_rows if _in_prefix(row, analysis_base)]
        selected = sorted(root_seed0 + analysis_seed12, key=_seed)
        if len(selected) == 3:
            return selected, "fixed_goal_multiseed"

    if block in {"B2", "B3", "B4"}:
        selected = sorted(
            [row for row in block_rows if _in_prefix(row, analysis_four)], key=_seed
        )
        if len(selected) == 3:
            label = {
                "B2": "fixed_goal_gold_standard",
                "B3": "fixed_goal_naive_ft",
                "B4": "fixed_goal_ga_refit",
            }[block]
            return selected, label

    if block == "B4T":
        selected = sorted(
            [row for row in block_rows if _in_prefix(row, root_prefix)], key=_seed
        )
        if len(selected) == 3:
            return selected, "trajdeleter_alpha1.0_beta2.0_s1100_s21000"

    return block_rows, None


def _select_canonical_b4t_rows(
    env: str, block_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    file_to_row = {str(row["file"]): row for row in block_rows}
    files = list(file_to_row)
    for selection, pattern in canonical_b4t_selection_specs(env):
        selected_files = [name for name in files if pattern.fullmatch(name)]
        if selected_files:
            return [file_to_row[name] for name in selected_files], selection
    return block_rows, None


def _summarize_block(
    rows: list[dict[str, Any]], env: str, block: str
) -> dict[str, Any]:
    block_rows = [row for row in rows if row["block"] == block]
    selection: str | None = None
    if block_rows:
        block_rows, selection = _select_antmaze_preferred_rows(env, block, block_rows)
    if block == "B4" and block_rows:
        block_rows, b4_selection = _select_canonical_b4_rows(env, block_rows)
        selection = selection or b4_selection
    if block == "B4T" and block_rows:
        block_rows, b4t_selection = _select_canonical_b4t_rows(env, block_rows)
        selection = selection or b4t_selection
    summary: dict[str, Any] = {
        "env": env,
        "block": block,
        "present": bool(block_rows),
        "n_files": len(block_rows),
        "provenance": sorted({str(row["provenance"]) for row in block_rows}),
        "files": [str(row["file"]) for row in block_rows],
        "paths": [str(row["path"]) for row in block_rows],
        "gate_status": _gate_status(block, block_rows),
    }

    if selection is not None:
        summary["selection"] = selection

    if block == "B6":
        by_model = defaultdict(list)
        for row in block_rows:
            by_model[str(row.get("model"))].append(row)
        summary["models"] = {
            model: sorted({str(item["provenance"]) for item in items})
            for model, items in sorted(by_model.items())
        }

    if block == "B7":
        ratio_map: dict[str, dict[str, list[str]]] = {}
        for ratio in B7_REQUIRED_RATIOS:
            ratio_key = f"{ratio:.2f}"
            ratio_rows = [
                row
                for row in block_rows
                if row.get("forget_ratio") is not None
                and float(row["forget_ratio"]) == ratio
            ]
            by_model = defaultdict(list)
            for row in ratio_rows:
                by_model[str(row.get("model"))].append(str(row["provenance"]))
            ratio_map[ratio_key] = {
                model: sorted(set(values)) for model, values in sorted(by_model.items())
            }
        summary["ratio_matrix"] = ratio_map

    return summary


def _env_status(block_summaries: list[dict[str, Any]], env: str) -> str:
    expected = set(EXPECTED_BLOCKS[env])
    present = {item["block"] for item in block_summaries if item["present"]}
    if expected - present:
        return "incomplete"

    for item in block_summaries:
        if item["block"] not in expected:
            continue
        provenances = set(item["provenance"])
        if "contract_placeholder" in provenances:
            return "validator_complete_with_placeholders"
        if item.get("gate_status") == "failed":
            return "real_runs_with_failed_gates"
    return "real_ready"


def _build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Audit",
        "",
        "This file is auto-generated by `scripts/build_benchmark_audit.py` to distinguish `validator complete` from `real-run provenance`.",
        "",
        "## Environment Summary",
        "",
        "| Environment | Validator | Audit Status | Notes |",
        "|---|---|---|---|",
    ]

    for env_summary in report["environments"]:
        notes = []
        for block in env_summary["blocks"]:
            if "contract_placeholder" in block["provenance"]:
                notes.append(f"{block['block']} contains placeholder artifacts")
            if block.get("gate_status") == "failed":
                notes.append(f"{block['block']} gate failed")
        note_text = "; ".join(notes) if notes else "Main blocks have real results"
        lines.append(
            f"| `{env_summary['env']}` | `{env_summary['validator_status']}` | `{env_summary['audit_status']}` | {note_text} |"
        )

    lines.extend(
        [
            "",
            "## Block Summary",
            "",
            "| Environment | Block | File Count | Provenance | Gate |",
            "|---|---|---:|---|---|",
        ]
    )

    for env_summary in report["environments"]:
        for block in env_summary["blocks"]:
            lines.append(
                f"| `{env_summary['env']}` | `{block['block']}` | {block['n_files']} | `{', '.join(block['provenance']) if block['provenance'] else 'missing'}` | `{block.get('gate_status', 'unknown')}` |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `real_ready`: All expected blocks present, no B6/B7 contract placeholders found, and no explicit gate failures in `B2/B3/B4/B4T`.",
            "- `real_runs_with_failed_gates`: All expected blocks present with real-run provenance, but at least one `B2/B3/B4/B4T` explicitly reports a gate failure.",
            "- `validator_complete_with_placeholders`: Validator passed, but at least one block is still a contract placeholder.",
            "- `canonical_export`: B1 back-filled via canonical exporter; valid benchmark asset but not an independently new training run.",
            "- `contract_placeholder`: Artifact satisfies the contract and validator, but values match the lightweight placeholder template and should not be cited as fully-real benchmark evidence.",
            "- `Gate`: Currently only consumes explicit gate fields from `B2/B3/B4/B4T` result JSONs; `failed` means at least one run reports `gold_standard_valid = false` or `retain_nll_shift_pass = false`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark results audit report")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_DIR),
        help="Results root directory, default: results",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Analysis output directory, default: results/analysis",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_json = output_dir / "benchmark_audit.json"
    output_csv = output_dir / "benchmark_audit.csv"
    output_md = output_dir / "benchmark_audit.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for env in TARGET_ENVS:
        env_dir = results_dir / env
        validator = validate_results_completeness(env_dir)
        rows = _discover_records(env_dir)
        block_summaries = [
            _summarize_block(rows, env, block) for block in EXPECTED_BLOCKS[env]
        ]
        audit_status = _env_status(block_summaries, env)

        env_reports.append(
            {
                "env": env,
                "validator_status": validator.status,
                "validator_errors": validator.errors,
                "validator_warnings": validator.warnings,
                "audit_status": audit_status,
                "blocks": block_summaries,
            }
        )

        for block in block_summaries:
            csv_rows.append(
                {
                    "env": env,
                    "validator_status": validator.status,
                    "audit_status": audit_status,
                    "block": block["block"],
                    "present": block["present"],
                    "n_files": block["n_files"],
                    "provenance": ";".join(block["provenance"]),
                    "gate_status": block.get("gate_status", "unknown"),
                    "files": ";".join(block["files"]),
                }
            )

    report = {"environments": env_reports}
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "env",
                "validator_status",
                "audit_status",
                "block",
                "present",
                "n_files",
                "provenance",
                "gate_status",
                "files",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    output_md.write_text(_build_markdown(report), encoding="utf-8")

    print(f"Saved audit JSON: {output_json}")
    print(f"Saved audit CSV: {output_csv}")
    print(f"Saved audit Markdown: {output_md}")


if __name__ == "__main__":
    main()
