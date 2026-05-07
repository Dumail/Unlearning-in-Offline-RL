from __future__ import annotations

import json
from pathlib import Path


B4_RETAIN_REQUIRED_FIELDS = (
    "retain_nll_baseline_mean",
    "retain_nll_current_mean",
    "retain_nll_shift_ratio",
    "retain_nll_shift_percent",
    "retain_nll_shift_pass",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0 if abs(numerator) < 1e-12 else float("inf")
    return float(numerator / denominator)


def compute_retain_shift_check(
    baseline_retain_nll_mean: float,
    current_retain_nll_mean: float,
    *,
    max_shift_ratio: float = 0.20,
) -> dict[str, float | bool]:
    baseline = float(baseline_retain_nll_mean)
    current = float(current_retain_nll_mean)
    shift_ratio = abs(_safe_ratio(current - baseline, baseline))
    return {
        "retain_nll_baseline_mean": baseline,
        "retain_nll_current_mean": current,
        "retain_nll_shift_ratio": shift_ratio,
        "retain_nll_shift_percent": shift_ratio * 100.0,
        "retain_nll_shift_pass": bool(shift_ratio < float(max_shift_ratio)),
    }


def load_base_retain_nll_mean(results_dir: str | Path, env_name: str) -> float | None:
    path = Path(results_dir) / env_name / "tmi_eval_dt_final.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("retain_nll_mean")
    if value is None:
        raise KeyError(f"{path} missing required key retain_nll_mean")
    return float(value)


def build_b4_retain_fields(
    tmi_results: dict,
    baseline_retain_nll_mean: float | None,
) -> dict[str, float | bool]:
    if "retain_nll_mean" not in tmi_results:
        raise KeyError("tmi_results missing required key retain_nll_mean")
    current = float(tmi_results["retain_nll_mean"])
    baseline = (
        current if baseline_retain_nll_mean is None else float(baseline_retain_nll_mean)
    )
    return compute_retain_shift_check(baseline, current)


def assert_b4_retain_schema(payload: dict) -> None:
    missing = [key for key in B4_RETAIN_REQUIRED_FIELDS if key not in payload]
    if missing:
        raise ValueError(f"missing required B4 retain field(s): {', '.join(missing)}")
