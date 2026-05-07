import csv
import json
import math
from pathlib import Path


FIG_DIR = Path(__file__).resolve().parent


def discover_root_dir(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "results").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root from {start}")


ROOT_DIR = discover_root_dir(FIG_DIR)
RESULTS_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"


APPENDIX_ENV_ORDER = [
    "hopper-medium-replay-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-replay-v2",
]

CANONICAL_ENV_ORDER = {
    "walker2d-medium-expert-v2": 0,
    "halfcheetah-medium-expert-v2": 1,
    "hopper-medium-expert-v2": 2,
    "walker2d-medium-v2": 3,
    "halfcheetah-medium-v2": 4,
    "hopper-medium-v2": 5,
    "walker2d-medium-replay-v2": 6,
    "halfcheetah-medium-replay-v2": 7,
    "hopper-medium-replay-v2": 8,
}

CANONICAL_ENV_LABELS = {
    "halfcheetah-medium-replay-v2": "HalfCheetah (R)",
    "hopper-medium-replay-v2": "Hopper (R)",
    "walker2d-medium-replay-v2": "Walker2D (R)",
    "halfcheetah-medium-v2": "HalfCheetah (M)",
    "hopper-medium-v2": "Hopper (M)",
    "walker2d-medium-v2": "Walker2D (M)",
    "halfcheetah-medium-expert-v2": "HalfCheetah (ME)",
    "hopper-medium-expert-v2": "Hopper (ME)",
    "walker2d-medium-expert-v2": "Walker2D (ME)",
}

APPENDIX_ENV_LABELS = {
    "hopper-medium-replay-v2": "Hopper",
    "halfcheetah-medium-replay-v2": "HalfCheetah",
    "walker2d-medium-replay-v2": "Walker2D",
}

HIGH_UTILITY_ATTACK_ENV_ORDER = [
    "halfcheetah-medium-expert-v2",
    "walker2d-medium-expert-v2",
    "walker2d-medium-v2",
]

HIGH_UTILITY_ATTACK_ENV_LABELS = {
    "halfcheetah-medium-expert-v2": "HalfCheetah (ME)",
    "walker2d-medium-expert-v2": "Walker2D (ME)",
    "walker2d-medium-v2": "Walker2D (M)",
}

ATTACK_METHOD_ORDER = [
    ("base_dt", "Base DT"),
    ("gold_standard", "Retrain Ref."),
    ("naive_ft", "Naive FT"),
    ("ga_refit", "GA+Refit"),
]

HIGH_UTILITY_ATTACK_METHOD_ORDER = [
    ("base_dt", "Base DT"),
    ("gold_standard", "Retrain Ref."),
    ("naive_ft", "Naive FT"),
    ("ga_refit", "GA+Refit"),
    ("trajdeleter", "TrajDeleter"),
]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_tex(name: str) -> str:
    return (FIG_DIR / name).read_text()


def maybe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    numeric = float(text)
    return None if math.isnan(numeric) else numeric


def maybe_int(value: str | float | int | None) -> int | None:
    numeric = maybe_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def format_score(value: float | None) -> str:
    return "---" if value is None else f"{value:.2f}"


def format_auc(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def format_gap(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def format_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "---"
    return f"[{low:.3f}, {high:.3f}]"


def format_ci_width(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def format_p_value(value: float | None) -> str:
    return "---" if value is None else f"{value:.4f}"


def format_gold_valid(value: bool | None) -> str:
    if value is None:
        return "---"
    return "Yes" if value else "No"


def format_gold_valid_summary(passes: int, n_seeds: int) -> str:
    if n_seeds <= 0:
        return "---"
    if passes <= 0:
        return "No"
    if passes == n_seeds:
        return "Yes"
    return f"{passes}/{n_seeds}"


def format_match_rate(left: int | None, right: int | None) -> str:
    if left is None or right is None:
        return "---"
    return f"{left}/{right}"


def format_transition(pre_value: float | None, post_value: float | None) -> str:
    if pre_value is None or post_value is None:
        return "---"
    return f"{pre_value:.3f} -> {post_value:.3f}"


def latex_row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def write_outputs(outputs: list[tuple[str, str]]) -> None:
    for name, content in outputs:
        path = FIG_DIR / name
        if not content.endswith("\n"):
            content = content + "\n"
        path.write_text(content)
        print(f"Saved: {path}")
