from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.tmi_multi_attack import multi_attack_evaluation

ENV_ALIASES = {
    "hopper": "hopper-medium-replay-v2",
    "halfcheetah": "halfcheetah-medium-replay-v2",
    "walker2d": "walker2d-medium-replay-v2",
}

METHOD_SEED_FILES = {
    "base_dt": lambda seed: "tmi_eval_dt_final.json"
    if seed == 0
    else f"tmi_eval_dt_seed{seed}.json",
    "gold_standard": lambda seed: f"gold_standard_seed{seed}.json",
    "naive_ft": lambda seed: f"naive_ft_seed{seed}.json",
    "ga_refit": lambda seed: f"ga_refit_lambda0.1_steps500_seed{seed}.json",
    "trajdeleter": lambda seed: f"trajdeleter_alpha1.0_beta2.0_s1100_s21000_seed{seed}.json",
}

SEEDS = [0, 1, 2]
BOOL_FIELDS = ["tost_equivalent", "all_gaps_below_margin"]
FLOAT_FIELDS = [
    "d4rl_score",
    "original_forget_auc",
    "nll_auc",
    "nll_gap",
    "threshold_balanced_acc",
    "threshold_gap",
    "znorm_auc",
    "znorm_gap",
    "variance_auc",
    "variance_gap",
    "reference_auc",
    "reference_gap",
    "tost_p_value",
    "max_gap",
]


def load_result_nlls(result_path: Path) -> dict | None:
    if not result_path.exists():
        return None
    with open(result_path) as f:
        data = json.load(f)
    if "forget_nlls" not in data or "negative_nlls" not in data:
        return None
    return {
        "forget_nlls": np.array(data["forget_nlls"]),
        "negative_nlls": np.array(data["negative_nlls"]),
        "forget_auc": data.get("forget_auc"),
        "d4rl_score": data.get("d4rl_score"),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["env"], row["method"]), []).append(row)

    summary_rows: list[dict] = []
    for (env_name, method_name), group in grouped.items():
        seeds = sorted(int(row["seed"]) for row in group)
        summary: dict[str, object] = {
            "env": env_name,
            "method": method_name,
            "n_seeds": len(group),
            "seed_list": ",".join(str(seed) for seed in seeds),
            "n_pairs": int(np.mean([int(row["n_pairs"]) for row in group])),
        }
        for field in FLOAT_FIELDS:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            summary[field] = float(np.mean(values)) if values else None
        for field in BOOL_FIELDS:
            values = [bool(row[field]) for row in group]
            summary[field] = all(values)
            summary[f"{field}_passes"] = int(sum(values))
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (row["env"], row["method"]))
    return summary_rows


def resolve_env_names(envs: list[str]) -> list[str]:
    resolved = []
    for env in envs:
        resolved.append(ENV_ALIASES.get(env, env))
    return resolved


def run_multi_attack_analysis(
    envs: list[str],
    results_dir: Path,
    output_dir: Path,
    margin: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    all_rows: list[dict] = []
    selected_envs = resolve_env_names(envs)

    for env_name in selected_envs:
        env_dir = results_dir / env_name
        if not env_dir.exists():
            print(f"[skip] {env_name}: results directory does not exist")
            continue

        for seed in SEEDS:
            gold_path = env_dir / METHOD_SEED_FILES["gold_standard"](seed)
            gold_data = load_result_nlls(gold_path)
            ref_forget = gold_data["forget_nlls"] if gold_data else None
            ref_negative = gold_data["negative_nlls"] if gold_data else None

            for method, filename_fn in METHOD_SEED_FILES.items():
                filepath = env_dir / filename_fn(seed)
                data = load_result_nlls(filepath)
                if data is None:
                    print(f"[skip] {env_name}/{method}/seed{seed}: no raw NLL data")
                    continue

                print(f"\n=== {env_name} / {method} / seed {seed} ===")
                print(
                    f"  forget={len(data['forget_nlls'])}, negative={len(data['negative_nlls'])}"
                )

                use_ref_forget = ref_forget if method != "gold_standard" else None
                use_ref_negative = ref_negative if method != "gold_standard" else None

                result = multi_attack_evaluation(
                    data["forget_nlls"],
                    data["negative_nlls"],
                    ref_forget_nlls=use_ref_forget,
                    ref_negative_nlls=use_ref_negative,
                    margin=margin,
                )

                row = {
                    "env": env_name,
                    "method": method,
                    "seed": seed,
                    "d4rl_score": data.get("d4rl_score"),
                    "original_forget_auc": data.get("forget_auc"),
                    **result,
                }
                all_rows.append(row)

    return all_rows, aggregate_rows(all_rows)


def save_results(rows: list[dict], summary_rows: list[dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    json_path = output_dir / "multi_attack_seed_rows.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=_convert)

    summary_json_path = output_dir / "multi_attack_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=_convert)

    seed_csv_fields = [
        "env",
        "method",
        "seed",
        "d4rl_score",
        "original_forget_auc",
        "n_pairs",
        "nll_auc",
        "nll_gap",
        "threshold_balanced_acc",
        "threshold_gap",
        "znorm_auc",
        "znorm_gap",
        "variance_auc",
        "variance_gap",
        "reference_auc",
        "reference_gap",
        "tost_equivalent",
        "tost_p_value",
        "all_gaps_below_margin",
        "max_gap",
    ]
    seed_csv_path = output_dir / "multi_attack_seed_rows.csv"
    with open(seed_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _convert(row.get(k, "")) for k in seed_csv_fields})

    summary_csv_fields = [
        "env",
        "method",
        "n_seeds",
        "seed_list",
        "n_pairs",
        "d4rl_score",
        "original_forget_auc",
        "nll_auc",
        "nll_gap",
        "threshold_balanced_acc",
        "threshold_gap",
        "znorm_auc",
        "znorm_gap",
        "variance_auc",
        "variance_gap",
        "reference_auc",
        "reference_gap",
        "tost_equivalent",
        "tost_equivalent_passes",
        "tost_p_value",
        "all_gaps_below_margin",
        "all_gaps_below_margin_passes",
        "max_gap",
    ]
    summary_csv_path = output_dir / "multi_attack_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: _convert(row.get(k, "")) for k in summary_csv_fields})

    md_path = output_dir / "multi_attack_summary.md"
    with open(md_path, "w") as f:
        f.write("# Multi-Attack Privacy Audit Summary\n\n")
        f.write("Seed-level rows are saved to `multi_attack_seed_rows.{json,csv}`.\n\n")
        f.write(
            "| Env | Method | Seeds | N | NLL AUC | Threshold | Z-norm | Variance | Reference | TOST pass | All<ε pass |\n"
        )
        f.write(
            "|-----|--------|-------|---|---------|-----------|--------|----------|-----------|-----------|-------------|\n"
        )
        for row in summary_rows:
            ref_auc = row.get("reference_auc")
            ref_str = f"{ref_auc:.3f}" if ref_auc is not None else "-"
            f.write(
                f"| {row['env'].split('-')[0]} | {row['method']} | {row['seed_list']} | {row['n_pairs']} "
                f"| {row['nll_auc']:.3f} | {row['threshold_balanced_acc']:.3f} | {row['znorm_auc']:.3f} "
                f"| {row['variance_auc']:.3f} | {ref_str} | {row['tost_equivalent_passes']}/{row['n_seeds']} "
                f"| {row['all_gaps_below_margin_passes']}/{row['n_seeds']} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Multi-attack privacy audit")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["hopper", "halfcheetah", "walker2d"],
        help="Environments to analyze",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis/multi_attack",
        help="Output directory",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="TOST equivalence test margin",
    )
    args = parser.parse_args()

    rows, summary_rows = run_multi_attack_analysis(
        envs=args.envs,
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        margin=args.margin,
    )

    if rows:
        save_results(rows, summary_rows, Path(args.output_dir))
        print(
            f"\nDone: {len(rows)} seed-level results, {len(summary_rows)} summary results"
        )
    else:
        print("\nNo results available")


if __name__ == "__main__":
    main()
