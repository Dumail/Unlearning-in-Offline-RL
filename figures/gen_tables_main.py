"""Generate 4 main-text tables as plain text (.txt) files.

Outputs:
- TABLE_1_benchmark.txt          tab:benchmark        9 locomotion settings
- TABLE_multi_attack.txt         tab:multi_attack     3 high-utility settings
- TABLE_architecture_summary.txt tab:architecture_summary  3 replay envs x 3 archs
- TABLE_2_oracle.txt             tab:oracle           3 replay envs selective held-out CV
"""

from __future__ import annotations

from table_utils import (
    ANALYSIS_DIR,
    APPENDIX_ENV_LABELS,
    APPENDIX_ENV_ORDER,
    CANONICAL_ENV_LABELS,
    CANONICAL_ENV_ORDER,
    HIGH_UTILITY_ATTACK_ENV_LABELS,
    HIGH_UTILITY_ATTACK_ENV_ORDER,
    RESULTS_DIR,
    format_auc,
    format_ci,
    format_ci_width,
    format_gap,
    format_gold_valid,
    format_score,
    load_csv_rows,
    load_json,
    maybe_float,
    maybe_int,
    write_outputs,
)


# ----------------------------- Common helpers -----------------------------

CANONICAL_METHOD_ORDER = [
    ("B1", "Base DT"),
    ("B2", "Retrain Ref."),
    ("B3", "Naive FT"),
    ("B4", "GA+Refit"),
]

LOCOMOTION_ENVS = {
    "halfcheetah-medium-replay-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2",
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-expert-v2",
}

MULTI_ATTACK_METHOD_ORDER = [
    ("base_dt", "Base DT"),
    ("gold_standard", "Retrain Ref."),
    ("naive_ft", "Naive FT"),
    ("ga_refit", "GA+Refit"),
    ("trajdeleter", "TrajDeleter"),
]


def summary_float(summary: dict[str, object], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (str, float, int)) or value is None:
        return maybe_float(value)
    return None


def summary_int(summary: dict[str, object], key: str) -> int | None:
    value = summary.get(key)
    if isinstance(value, (str, float, int)) or value is None:
        return maybe_int(value)
    return None


def auc_to_gap(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value - 0.5)


def format_ratio(value: float | None) -> str:
    return "---" if value is None else f"{value:.2f}"


def require_three_seed_summary(
    summary_rows: dict[tuple[str, str], dict[str, object]],
    env_name: str,
    block_name: str,
) -> dict[str, object]:
    summary = summary_rows.get((env_name, block_name))
    if summary is None:
        raise ValueError(
            f"TABLE_1 requires a canonical multiseed summary for {env_name} {block_name}; fallback is disabled."
        )
    n_seeds = summary_int(summary, "n_seeds")
    if n_seeds != 3:
        raise ValueError(
            f"TABLE_1 requires exactly 3 seeds for {env_name} {block_name}, got {n_seeds}."
        )
    return summary


def read_mlp_d4rl_from_json(env_name: str) -> float | None:
    path = RESULTS_DIR / env_name / "b6_mlp_summary.json"
    if not path.exists():
        return None
    payload = load_json(path)
    return maybe_float(payload.get("d4rl_score"))


# ----------------------------- Plain text table helpers -----------------------------


def _text_row(cells: list[str], widths: list[int], pad: str = " ") -> str:
    """Format a single row with fixed-width columns."""
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(cell.rjust(w) if cell else " " * w)
    return "  ".join(parts)


def _text_separator(widths: list[int], char: str = "-") -> str:
    """Generate a separator line."""
    parts = [char * w for w in widths]
    return "  ".join(parts)


def _build_text_table(
    title: str,
    headers: list[str],
    widths: list[int],
    rows: list[list[str]],
    separator_indices: set[int] | None = None,
) -> str:
    """Build a complete plain text table."""
    lines = [title, ""]
    lines.append(_text_row(headers, widths))
    lines.append(_text_separator(widths, "="))
    for i, row in enumerate(rows):
        lines.append(_text_row(row, widths))
        if separator_indices and i in separator_indices:
            lines.append(_text_separator(widths, "-"))
    lines.append(_text_separator(widths, "="))
    return "\n".join(lines)


def _mark_bold(text: str, is_bold: bool) -> str:
    """Mark text as bold with ** markers for plain text."""
    if is_bold:
        return f"*{text}*"
    return text


# ----------------------------- TABLE_1_benchmark -----------------------------


def build_table1() -> str:
    audit = load_json(ANALYSIS_DIR / "benchmark_audit.json")
    multiseed = load_json(ANALYSIS_DIR / "b1_b4_multiseed_summary.json")
    summary_rows = {
        (row["env"], row["block"]): row for row in multiseed.get("rows", [])
    }

    headers = [
        "Environment",
        "Method",
        "Utility\u2191",
        "Forget Gap\u2193",
        "Forget AUC",
        "95% CI",
        "Retain AUC",
    ]
    widths = [16, 14, 12, 14, 12, 20, 12]

    environments = sorted(
        audit.get("environments", []),
        key=lambda item: CANONICAL_ENV_ORDER.get(item["env"], 999),
    )
    environments = [item for item in environments if item["env"] in LOCOMOTION_ENVS]

    all_rows: list[list[str]] = []
    separator_indices: set[int] = set()

    for env_index, env_entry in enumerate(environments):
        env_name = env_entry["env"]
        env_label = CANONICAL_ENV_LABELS.get(env_name, env_name)
        block_map = {block["block"]: block for block in env_entry.get("blocks", [])}
        rendered_rows: list[tuple[str, str, str, str, str, str]] = []
        raw_gaps: list[float | None] = []

        for block_name, method_label in CANONICAL_METHOD_ORDER:
            block = block_map.get(block_name, {})
            if not block.get("present", False):
                raise ValueError(
                    f"TABLE_1 requires {env_name} {block_name} to be present in benchmark_audit.json."
                )
            summary = require_three_seed_summary(summary_rows, env_name, block_name)
            gap_raw = auc_to_gap(summary_float(summary, "forget_auc_mean"))
            raw_gaps.append(gap_raw)
            rendered_rows.append(
                (
                    method_label,
                    format_score(summary_float(summary, "d4rl_score_mean")),
                    format_gap(gap_raw),
                    format_auc(summary_float(summary, "forget_auc_mean")),
                    format_ci(
                        summary_float(summary, "forget_auc_seed_ci_low"),
                        summary_float(summary, "forget_auc_seed_ci_high"),
                    ),
                    format_auc(summary_float(summary, "retain_diag_auc_mean")),
                )
            )

        b4t_block = block_map.get("B4T", {})
        if b4t_block.get("present", False):
            summary = require_three_seed_summary(summary_rows, env_name, "B4T")
            gap_raw_b4t = auc_to_gap(summary_float(summary, "forget_auc_mean"))
            raw_gaps.append(gap_raw_b4t)
            rendered_rows.append(
                (
                    "TrajDeleter",
                    format_score(summary_float(summary, "d4rl_score_mean")),
                    format_gap(gap_raw_b4t),
                    format_auc(summary_float(summary, "forget_auc_mean")),
                    format_ci(
                        summary_float(summary, "forget_auc_seed_ci_low"),
                        summary_float(summary, "forget_auc_seed_ci_high"),
                    ),
                    format_auc(summary_float(summary, "retain_diag_auc_mean")),
                )
            )

        # Find the row with the smallest Forget Gap in each environment
        min_gap_idx = min(
            (i for i, g in enumerate(raw_gaps) if g is not None),
            key=lambda i: raw_gaps[i],  # type: ignore[arg-type]
            default=-1,
        )

        for row_idx, (
            method_label,
            d4rl_score,
            forget_gap,
            forget_auc,
            ci,
            retain_auc,
        ) in enumerate(rendered_rows):
            gap_display = _mark_bold(forget_gap, row_idx == min_gap_idx)
            env_display = env_label if row_idx == 0 else ""
            all_rows.append(
                [
                    env_display,
                    method_label,
                    d4rl_score,
                    gap_display,
                    forget_auc,
                    ci,
                    retain_auc,
                ]
            )

        if env_index != len(environments) - 1:
            separator_indices.add(len(all_rows) - 1)
        if env_index < len(environments) - 1 and (env_index + 1) % 3 == 0:
            separator_indices.add(len(all_rows) - 1)

    title = (
        "Table 1: Benchmark results across R/M/ME variants of three locomotion environments.\n"
        "Utility = D4RL normalized score. Forget Gap = |Forget AUC - 0.5|.\n"
        "Best Forget Gap per environment is marked with *."
    )
    return _build_text_table(title, headers, widths, all_rows, separator_indices)


# ----------------------------- TABLE_multi_attack -----------------------------


def build_table_multi_attack() -> str:
    rows_by_key = {
        (row["env"], row["method"]): row
        for row in load_csv_rows(
            ANALYSIS_DIR / "multi_attack_high_utility" / "multi_attack_summary.csv"
        )
    }

    headers = [
        "Environment",
        "Method",
        "Pairs",
        "NLL AUC",
        "Thr BA",
        "Ref AUC",
        "Dev AUC",
        "TOST",
        "All<eps",
    ]
    widths = [16, 14, 8, 10, 10, 10, 10, 8, 10]

    def render_reference_auc(row: dict[str, str]) -> str:
        value = maybe_float(row.get("reference_auc"))
        if value is None:
            return "---"
        return format_auc(value)

    def closest_to_half_index(values: list[float | None]) -> int:
        best_idx = -1
        best_dist = float("inf")
        for i, v in enumerate(values):
            if v is not None:
                dist = abs(v - 0.5)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        return best_idx

    all_rows: list[list[str]] = []
    separator_indices: set[int] = set()

    for env_index, env_name in enumerate(HIGH_UTILITY_ATTACK_ENV_ORDER):
        env_label = HIGH_UTILITY_ATTACK_ENV_LABELS[env_name]
        rendered_rows = []
        raw_nll: list[float | None] = []
        raw_thr: list[float | None] = []
        raw_ref: list[float | None] = []
        raw_dev: list[float | None] = []

        for method_key, method_label in MULTI_ATTACK_METHOD_ORDER:
            row = rows_by_key[(env_name, method_key)]
            nll_val = maybe_float(row.get("nll_auc"))
            thr_val = maybe_float(row.get("threshold_balanced_acc"))
            ref_val = maybe_float(row.get("reference_auc"))
            dev_val = maybe_float(row.get("variance_auc"))
            raw_nll.append(nll_val)
            raw_thr.append(thr_val)
            raw_ref.append(ref_val)
            raw_dev.append(dev_val)
            rendered_rows.append(
                {
                    "method": "Retrain Ref."
                    if method_key == "gold_standard"
                    else method_label,
                    "pairs": str(maybe_int(row.get("n_pairs")) or "---"),
                    "nll_auc": format_auc(nll_val),
                    "thr_ba": format_auc(thr_val),
                    "ref_auc": render_reference_auc(row),
                    "var_auc": format_auc(dev_val),
                    "tost": format_gold_valid(
                        str(row.get("tost_equivalent", "")).lower() == "true"
                    ),
                    "all_below": format_gold_valid(
                        str(row.get("all_gaps_below_margin", "")).lower() == "true"
                    ),
                }
            )

        bold_map = {
            "nll_auc": closest_to_half_index(raw_nll),
            "thr_ba": closest_to_half_index(raw_thr),
            "ref_auc": closest_to_half_index(raw_ref),
            "var_auc": closest_to_half_index(raw_dev),
        }
        for i, row in enumerate(rendered_rows):
            for col in ("nll_auc", "thr_ba", "ref_auc", "var_auc"):
                if i == bold_map[col] and row[col] != "---":
                    row[col] = _mark_bold(row[col], True)

        for row_idx, row in enumerate(rendered_rows):
            env_display = env_label if row_idx == 0 else ""
            all_rows.append(
                [
                    env_display,
                    row["method"],
                    row["pairs"],
                    row["nll_auc"],
                    row["thr_ba"],
                    row["ref_auc"],
                    row["var_auc"],
                    row["tost"],
                    row["all_below"],
                ]
            )

        if env_index != len(HIGH_UTILITY_ATTACK_ENV_ORDER) - 1:
            separator_indices.add(len(all_rows) - 1)

    title = (
        "Table: Multi-attack privacy audit on selected high-utility settings (aggregated across seeds).\n"
        "NLL: per-timestep NLL AUC. Thr: threshold MIA balanced accuracy.\n"
        "Ref: reference-model calibrated AUC. Dev: NLL-deviation AUC.\n"
        "Values closer to 0.5 indicate weaker membership evidence.\n"
        "TOST: equivalence test at alpha=0.05. All<eps: all gaps < eps=0.1."
    )
    return _build_text_table(title, headers, widths, all_rows, separator_indices)


# ----------------------------- TABLE_architecture_summary -----------------------------


def build_table_architecture_summary() -> str:
    b6_rows_by_key = {
        (row["env"], row["model"]): row
        for row in load_csv_rows(ANALYSIS_DIR / "b6_matched_budget_multiseed.csv")
    }
    env_order = [
        "hopper-medium-replay-v2",
        "halfcheetah-medium-replay-v2",
        "walker2d-medium-replay-v2",
    ]
    model_order = [("dt", "DT"), ("mlp", "MLP"), ("lstm", "LSTM")]

    headers = [
        "Environment",
        "Model",
        "Params",
        "Utility\u2191",
        "Forget Gap\u2193",
        "CI Width",
    ]
    widths = [16, 8, 10, 12, 14, 12]

    all_rows: list[list[str]] = []
    separator_indices: set[int] = set()

    for env_index, env_name in enumerate(env_order):
        env_label = APPENDIX_ENV_LABELS[env_name]
        rendered_rows: list[tuple[str, str, str, str, str]] = []
        raw_gaps: list[float | None] = []
        for model_key, model_label in model_order:
            row = b6_rows_by_key[(env_name, model_key)]
            d4rl_score = maybe_float(row.get("d4rl_score"))
            if model_key == "mlp" and d4rl_score is None:
                d4rl_score = read_mlp_d4rl_from_json(env_name)
            forget_auc = maybe_float(row.get("forget_auc"))
            ci_low = maybe_float(row.get("forget_auc_ci_low"))
            ci_high = maybe_float(row.get("forget_auc_ci_high"))
            n_params = maybe_int(row.get("n_params"))
            params = f"{n_params // 1000}K" if n_params is not None else "---"
            ci_width = None
            if ci_low is not None and ci_high is not None:
                ci_width = round(ci_high, 3) - round(ci_low, 3)
            gap_raw = auc_to_gap(forget_auc)
            raw_gaps.append(gap_raw)
            rendered_rows.append(
                (
                    model_label,
                    params,
                    format_score(d4rl_score),
                    format_gap(gap_raw),
                    format_ci_width(ci_width),
                )
            )

        min_gap_idx = min(
            (i for i, g in enumerate(raw_gaps) if g is not None),
            key=lambda i: raw_gaps[i],  # type: ignore[arg-type]
            default=-1,
        )

        for row_idx, (
            model_label,
            params,
            d4rl_score,
            forget_gap,
            ci_width,
        ) in enumerate(rendered_rows):
            gap_display = _mark_bold(forget_gap, row_idx == min_gap_idx)
            env_display = env_label if row_idx == 0 else ""
            all_rows.append(
                [
                    env_display,
                    model_label,
                    params,
                    d4rl_score,
                    gap_display,
                    ci_width,
                ]
            )

        if env_index != len(env_order) - 1:
            separator_indices.add(len(all_rows) - 1)

    title = (
        "Table: Architecture diagnostic for replay variants under unified backbone comparison.\n"
        "All rows use backbone-policy evaluations with matched controls and TMI audit.\n"
        "Best Forget Gap per environment is marked with *."
    )
    return _build_text_table(title, headers, widths, all_rows, separator_indices)


# ----------------------------- TABLE_2_oracle -----------------------------


def build_table2() -> str:
    budget_rows = {
        row["env"]: row
        for row in load_csv_rows(
            ANALYSIS_DIR
            / "selective_utility_budget"
            / "selective_utility_budget_summary.csv"
        )
        if maybe_float(row.get("budget")) == 0.0
    }
    multiseed = load_json(ANALYSIS_DIR / "b1_b4_multiseed_summary.json")
    base_rows = {
        row["env"]: row
        for row in multiseed.get("rows", [])
        if row.get("block") == "B1" and maybe_int(row.get("n_seeds")) == 3
    }
    gold_rows = {
        row["env"]: row
        for row in multiseed.get("rows", [])
        if row.get("block") == "B2" and maybe_int(row.get("n_seeds")) == 3
    }

    headers = [
        "Environment",
        "Method",
        "Budget OK",
        "Forget Gap\u2193",
        "Utility\u2191",
        "Score/Retrain",
        "Score/Base",
    ]
    widths = [16, 22, 10, 14, 12, 14, 12]

    all_rows: list[list[str]] = []
    separator_indices: set[int] = set()

    for env_index, env_name in enumerate(APPENDIX_ENV_ORDER):
        env_label = APPENDIX_ENV_LABELS[env_name]
        base_row = base_rows[env_name]
        gold_row = gold_rows[env_name]
        budget_row = budget_rows[env_name]
        base_score = summary_float(base_row, "d4rl_score_mean")
        gold_score = summary_float(gold_row, "d4rl_score_mean")
        gold_gap = None
        gold_auc = summary_float(gold_row, "forget_auc_mean")
        if gold_auc is not None:
            gold_gap = abs(gold_auc - 0.5)

        def ratio(score: float | None, anchor: float | None) -> float | None:
            if score is None or anchor is None or abs(anchor) <= 1e-12:
                return None
            return score / anchor

        selective_score = maybe_float(budget_row.get("selected_d4rl_mean"))
        uniform_score = maybe_float(budget_row.get("uniform_d4rl_mean"))

        rendered_rows: list[dict[str, str | float | None]] = [
            {
                "method": "Retraining reference",
                "success": "---",
                "gap": gold_gap,
                "score": gold_score,
                "score_gold_ratio": ratio(gold_score, gold_score),
                "score_base_ratio": ratio(gold_score, base_score),
            },
            {
                "method": "Selective (held-out CV)",
                "success": (
                    f"{maybe_int(budget_row.get('n_success'))}/{maybe_int(budget_row.get('n_total'))}"
                ),
                "gap": maybe_float(budget_row.get("selected_gap_mean")),
                "score": selective_score,
                "score_gold_ratio": ratio(selective_score, gold_score),
                "score_base_ratio": ratio(selective_score, base_score),
            },
            {
                "method": "Uniform (matched baseline)",
                "success": "---",
                "gap": maybe_float(budget_row.get("uniform_gap_mean")),
                "score": uniform_score,
                "score_gold_ratio": ratio(uniform_score, gold_score),
                "score_base_ratio": ratio(uniform_score, base_score),
            },
        ]

        gap_values = [maybe_float(r["gap"]) for r in rendered_rows]
        min_gap_idx = min(
            (i for i, g in enumerate(gap_values) if g is not None),
            key=lambda i: gap_values[i],  # type: ignore[arg-type]
            default=-1,
        )

        for row_idx, row in enumerate(rendered_rows):
            method_text = _mark_bold(str(row["method"]), row_idx == min_gap_idx)
            gap_text = _mark_bold(
                format_gap(maybe_float(row["gap"])), row_idx == min_gap_idx
            )
            env_display = env_label if row_idx == 0 else ""
            all_rows.append(
                [
                    env_display,
                    method_text,
                    str(row["success"]),
                    gap_text,
                    format_score(maybe_float(row["score"])),
                    format_ratio(maybe_float(row["score_gold_ratio"])),
                    format_ratio(maybe_float(row["score_base_ratio"])),
                ]
            )

        if env_index != len(APPENDIX_ENV_ORDER) - 1:
            separator_indices.add(len(all_rows) - 1)

    title = (
        "Table: Component-level GA under zero additional utility loss budget.\n"
        "Score/Retrain and Score/Base report retained-performance ratios.\n"
        "Best Forget Gap per environment is marked with *."
    )
    return _build_text_table(title, headers, widths, all_rows, separator_indices)


# ----------------------------- Main entry point -----------------------------


def generate_main_tables() -> list[tuple[str, str]]:
    return [
        ("TABLE_1_benchmark.txt", build_table1()),
        ("TABLE_multi_attack.txt", build_table_multi_attack()),
        ("TABLE_architecture_summary.txt", build_table_architecture_summary()),
        ("TABLE_2_oracle.txt", build_table2()),
    ]


if __name__ == "__main__":
    write_outputs(generate_main_tables())
