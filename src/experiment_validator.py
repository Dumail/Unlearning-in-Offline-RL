from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .experiment_contract import (
    B7_REQUIRED_RATIOS,
    BLOCK_CONTRACTS,
    OPTIONAL_BACKBONE_MODELS,
    SUPPORTED_MODELS,
    expected_blocks_for_env,
    normalize_ratio,
)


JSON_SUFFIX = ".json"
MODEL_PATTERN = re.compile(r"(?:^|[_\-])(dt|mlp|lstm|tt)(?:[_\-.]|$)", re.IGNORECASE)
RATIO_PATTERN = re.compile(r"ratio([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
SEED_PATTERN = re.compile(r"seed([0-9]+)", re.IGNORECASE)
ANALYSIS_JSON_PATTERNS = (
    re.compile(r"ablation_summary_seed\d+\.json$", re.IGNORECASE),
    re.compile(r"base_dt_tmi_3seed\.json$", re.IGNORECASE),
    re.compile(r"ft_step_decay\.json$", re.IGNORECASE),
)


def related_results_dirs(results_dir: Path) -> list[Path]:
    if not results_dir.exists() and not results_dir.parent.exists():
        return [results_dir]

    dirs: list[Path] = [results_dir]
    env_name = results_dir.name.strip().lower()
    if not env_name.startswith("antmaze-"):
        return dirs

    analysis_dir = results_dir.parent / "analysis"
    extra_dirs = [
        analysis_dir / "antmaze_fixed_goal_base_multiseed" / env_name,
        analysis_dir / "antmaze_four_method_runs" / env_name,
    ]
    for extra_dir in extra_dirs:
        if extra_dir.exists():
            dirs.append(extra_dir)
    return dirs


@dataclass
class ValidationResult:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ArtifactRecord:
    path: Path
    block: str | None
    payload: dict


def _discover_json_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        p
        for p in results_dir.rglob(f"*{JSON_SUFFIX}")
        if p.is_file()
        and not any(pattern.search(p.name) for pattern in ANALYSIS_JSON_PATTERNS)
    )


def _infer_block(path: Path, payload: dict) -> str | None:
    block = payload.get("block")
    if isinstance(block, str) and block in BLOCK_CONTRACTS:
        return block

    name = path.name.lower()
    parent = str(path.parent).lower()
    if name.startswith("gold_standard_seed"):
        return "B2"
    if name.startswith("naive_ft_seed"):
        return "B3"
    if name.startswith("ga_refit_"):
        return "B4"
    if name.startswith("trajdeleter_"):
        return "B4T"
    if (
        "ablation" in parent
        and name.startswith("lambda")
        and name.endswith(JSON_SUFFIX)
    ):
        return "B5"
    if name.startswith("b6_"):
        return "B6"
    if name.startswith("b7_"):
        return "B7"
    if name.startswith("base_") or name.startswith("tmi_eval_dt_final"):
        return "B1"
    if re.match(r"tmi_eval_dt_seed\d+\.json$", name):
        return "B1"
    return None


def _read_records(results_dir: Path, errors: list[str]) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for source_dir in related_results_dirs(results_dir):
        for path in _discover_json_files(source_dir):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    errors.append(f"{path}: invalid JSON schema, root must be object")
                    continue
            except Exception as exc:
                errors.append(f"{path}: invalid JSON parse error: {exc}")
                continue
            records.append(
                ArtifactRecord(
                    path=path, block=_infer_block(path, payload), payload=payload
                )
            )
    return records


def _validate_required_keys(records: list[ArtifactRecord], errors: list[str]) -> None:
    for record in records:
        if record.block is None:
            continue
        contract = BLOCK_CONTRACTS[record.block]
        missing = [k for k in contract.required_keys if k not in record.payload]
        if missing:
            missing_str = ", ".join(missing)
            errors.append(
                f"{record.path}: missing required key(s) [{missing_str}] for {record.block}"
            )


def _extract_model(record: ArtifactRecord) -> str | None:
    raw_model = record.payload.get("model")
    if isinstance(raw_model, str):
        model = raw_model.lower().strip()
        if model:
            return model
    matched = MODEL_PATTERN.search(record.path.name)
    if matched:
        return matched.group(1).lower()
    return None


def _extract_ratio(record: ArtifactRecord) -> float | None:
    raw_ratio = record.payload.get("forget_ratio")
    if raw_ratio is not None:
        try:
            return normalize_ratio(raw_ratio)
        except Exception:
            return None
    matched = RATIO_PATTERN.search(record.path.name)
    if matched:
        return normalize_ratio(matched.group(1))
    return None


def _extract_seed(record: ArtifactRecord) -> int | None:
    raw_seed = record.payload.get("seed")
    if raw_seed is not None:
        try:
            return int(raw_seed)
        except Exception:
            return None
    matched = SEED_PATTERN.search(record.path.name)
    if matched:
        return int(matched.group(1))
    return None


def _fmt_ratios(values: list[float] | set[float] | tuple[float, ...]) -> list[str]:
    return [f"{float(v):.2f}" for v in sorted(values)]


def _validate_b6_b7_dimensions(
    records: list[ArtifactRecord], errors: list[str]
) -> None:
    b6 = [r for r in records if r.block == "B6"]
    b7 = [r for r in records if r.block == "B7"]

    b6_models: set[str] = set()
    valid_b6_models = set(SUPPORTED_MODELS) | set(OPTIONAL_BACKBONE_MODELS)
    for record in b6:
        model = _extract_model(record)
        if model is None:
            errors.append(
                f"{record.path}: missing dimension model in naming/metadata for B6"
            )
            continue
        if model not in valid_b6_models:
            errors.append(
                f"{record.path}: invalid model '{model}', expected one of {sorted(valid_b6_models)}"
            )
            continue
        b6_models.add(model)

    expected_models: set[str] = set(SUPPORTED_MODELS)
    if "lstm" in b6_models:
        expected_models.add("lstm")
    if b6 and b6_models != expected_models:
        missing_models = sorted(expected_models - b6_models)
        errors.append(
            "B6 incomplete model dimension: "
            f"missing model={missing_models}, observed={sorted(b6_models)}"
        )

    b7_map: dict[str, set[float]] = {}
    for record in b7:
        model = _extract_model(record)
        ratio = _extract_ratio(record)
        if model is None:
            errors.append(
                f"{record.path}: missing dimension model in naming/metadata for B7"
            )
            continue
        if model not in valid_b6_models:
            errors.append(
                f"{record.path}: invalid model '{model}', expected one of {sorted(valid_b6_models)}"
            )
            continue
        if ratio is None:
            errors.append(
                f"{record.path}: missing dimension forget_ratio in naming/metadata for B7"
            )
            continue
        expected = set(B7_REQUIRED_RATIOS)
        if ratio not in expected:
            errors.append(
                f"{record.path}: invalid forget_ratio={ratio}, expected one of {sorted(expected)}"
            )
            continue
        b7_map.setdefault(model, set()).add(ratio)

    if b7:
        expected = set(B7_REQUIRED_RATIOS)
        for model, ratios in sorted(b7_map.items()):
            missing = sorted(expected - ratios)
            extra = sorted(ratios - expected)
            if missing or extra:
                errors.append(
                    "B7 incomplete forget_ratio dimension: "
                    f"model={model}, missing={_fmt_ratios(missing)}, "
                    f"extra={_fmt_ratios(extra)}, expected={_fmt_ratios(expected)}"
                )


def _validate_ambiguous_collisions(
    records: list[ArtifactRecord], errors: list[str]
) -> None:
    b7_records = [r for r in records if r.block == "B7"]
    key_to_path: dict[tuple[str, float, int], Path] = {}
    for record in b7_records:
        model = _extract_model(record)
        ratio = _extract_ratio(record)
        seed = _extract_seed(record)
        if model is None or ratio is None or seed is None:
            continue
        key = (model, ratio, seed)
        if key in key_to_path:
            errors.append(
                "legacy/ambiguous naming collision: "
                f"{record.path} conflicts with {key_to_path[key]} "
                f"on dimension model={model}, forget_ratio={ratio}, seed={seed}"
            )
        else:
            key_to_path[key] = record.path

    b6_records = [r for r in records if r.block == "B6"]
    b6_key_to_path: dict[str, Path] = {}
    for record in b6_records:
        model = _extract_model(record)
        if model is None:
            continue
        if model in b6_key_to_path:
            errors.append(
                "legacy/ambiguous naming collision: "
                f"{record.path} conflicts with {b6_key_to_path[model]} on dimension model={model}"
            )
        else:
            b6_key_to_path[model] = record.path


def _validate_block_presence(
    records: list[ArtifactRecord], results_dir: Path, errors: list[str]
) -> None:
    env_names = {
        str(env).strip().lower()
        for record in records
        for env in [record.payload.get("env")]
        if isinstance(env, str) and env.strip()
    }
    if not env_names:
        env_names.add(results_dir.name.strip().lower())

    seen_blocks = {r.block for r in records if r.block is not None}
    expected_blocks = {
        block for env_name in env_names for block in expected_blocks_for_env(env_name)
    }
    for block in BLOCK_CONTRACTS:
        if block not in expected_blocks:
            continue
        if block not in seen_blocks:
            errors.append(
                f"missing required artifact block={block} (no JSON matched contract hints {BLOCK_CONTRACTS[block].file_hints})"
            )


def validate_results_completeness(results_dir: str | Path) -> ValidationResult:
    path = Path(results_dir)
    errors: list[str] = []
    warnings: list[str] = []

    records = _read_records(path, errors)
    _validate_required_keys(records, errors)
    _validate_b6_b7_dimensions(records, errors)
    _validate_ambiguous_collisions(records, errors)
    _validate_block_presence(records, path, errors)

    if errors:
        schema_invalid = any(
            "invalid JSON" in e or "missing required key" in e for e in errors
        )
        status = "invalid" if schema_invalid else "incomplete"
        return ValidationResult(status=status, errors=errors, warnings=warnings)
    return ValidationResult(status="complete", errors=errors, warnings=warnings)
