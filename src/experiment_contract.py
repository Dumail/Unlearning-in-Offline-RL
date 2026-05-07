from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple, Pattern

from .retain_checks import B4_RETAIN_REQUIRED_FIELDS


TMI_REQUIRED_KEYS = {
    "env",
    "d4rl_score",
    "forget_auc",
    "forget_auc_ci_low",
    "forget_auc_ci_high",
    "retain_diag_auc",
}

B7_REQUIRED_RATIOS = (0.05, 0.10, 0.20)
SUPPORTED_MODELS = ("dt", "mlp")
OPTIONAL_BACKBONE_MODELS = ("lstm", "tt")


class BenchmarkEnvSpec(NamedTuple):
    key: str
    hydra_env: str
    env_name: str
    variant: str


CANONICAL_ENV_SPECS = (
    BenchmarkEnvSpec(
        "halfcheetah", "halfcheetah_mr", "halfcheetah-medium-replay-v2", "replay"
    ),
    BenchmarkEnvSpec("hopper", "hopper_mr", "hopper-medium-replay-v2", "replay"),
    BenchmarkEnvSpec("walker2d", "walker2d_mr", "walker2d-medium-replay-v2", "replay"),
    BenchmarkEnvSpec("halfcheetah", "halfcheetah_m", "halfcheetah-medium-v2", "medium"),
    BenchmarkEnvSpec("hopper", "hopper_m", "hopper-medium-v2", "medium"),
    BenchmarkEnvSpec("walker2d", "walker2d_m", "walker2d-medium-v2", "medium"),
    BenchmarkEnvSpec(
        "halfcheetah",
        "halfcheetah_me",
        "halfcheetah-medium-expert-v2",
        "medium_expert",
    ),
    BenchmarkEnvSpec("hopper", "hopper_me", "hopper-medium-expert-v2", "medium_expert"),
    BenchmarkEnvSpec(
        "walker2d",
        "walker2d_me",
        "walker2d-medium-expert-v2",
        "medium_expert",
    ),
    BenchmarkEnvSpec("antmaze", "antmaze_umaze", "antmaze-umaze-v2", "antmaze"),
    BenchmarkEnvSpec(
        "antmaze",
        "antmaze_umaze_diverse",
        "antmaze-umaze-diverse-v2",
        "antmaze",
    ),
    BenchmarkEnvSpec(
        "antmaze",
        "antmaze_medium_diverse",
        "antmaze-medium-diverse-v2",
        "antmaze",
    ),
)

CANONICAL_VARIANTS = tuple(dict.fromkeys(spec.variant for spec in CANONICAL_ENV_SPECS))

DEFAULT_B4_CANONICAL_SELECTION_SPECS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "ga_refit_lambda0.1_steps500",
        re.compile(r"ga_refit_lambda0\.1_steps500_seed\d+\.json$"),
    ),
    (
        "ga_refit_lambda1.0_steps500",
        re.compile(r"ga_refit_lambda1\.0_steps500_seed\d+\.json$"),
    ),
    (
        "ga_refit_lambda10.0_steps500",
        re.compile(r"ga_refit_lambda10\.0_steps500_seed\d+\.json$"),
    ),
)

DEFAULT_B4T_CANONICAL_SELECTION_SPECS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "trajdeleter_alpha1.0_beta2.0_s1100_s21000",
        re.compile(r"trajdeleter_alpha1\.0_beta2\.0_s1100_s21000_seed\d+\.json$"),
    ),
)

ENV_B4_CANONICAL_SELECTION_SPECS: dict[str, tuple[tuple[str, Pattern[str]], ...]] = {
    "antmaze-umaze-v2": (
        (
            "ga_refit_lambda0.1_steps5_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps5_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps8_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps8_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps10_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps10_seed\d+_ascentlr1em05\.json$"),
        ),
        *DEFAULT_B4_CANONICAL_SELECTION_SPECS,
    ),
    "antmaze-umaze-diverse-v2": (
        (
            "ga_refit_lambda0.1_steps5_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps5_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps8_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps8_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps10_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps10_seed\d+_ascentlr1em05\.json$"),
        ),
        *DEFAULT_B4_CANONICAL_SELECTION_SPECS,
    ),
    "antmaze-medium-diverse-v2": (
        (
            "ga_refit_lambda0.1_steps5_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps5_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps8_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps8_seed\d+_ascentlr1em05\.json$"),
        ),
        (
            "ga_refit_lambda0.1_steps10_ascentlr1em05",
            re.compile(r"ga_refit_lambda0\.1_steps10_seed\d+_ascentlr1em05\.json$"),
        ),
        *DEFAULT_B4_CANONICAL_SELECTION_SPECS,
    ),
}

EXPECTED_BLOCKS_BY_ENV = {
    "halfcheetah-medium-replay-v2": (
        "B1",
        "B2",
        "B3",
        "B4",
        "B4T",
        "B5",
        "B6",
        "B7",
    ),
    "hopper-medium-replay-v2": ("B1", "B2", "B3", "B4", "B4T", "B6", "B7"),
    "walker2d-medium-replay-v2": (
        "B1",
        "B2",
        "B3",
        "B4",
        "B4T",
        "B6",
        "B7",
    ),
    "halfcheetah-medium-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "hopper-medium-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "walker2d-medium-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "halfcheetah-medium-expert-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "hopper-medium-expert-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "walker2d-medium-expert-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "antmaze-umaze-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "antmaze-umaze-diverse-v2": ("B1", "B2", "B3", "B4", "B4T"),
    "antmaze-medium-diverse-v2": ("B1", "B2", "B3", "B4", "B4T"),
}


@dataclass(frozen=True)
class BlockContract:
    block_id: str
    required_keys: tuple[str, ...]
    required_dimensions: tuple[str, ...]
    file_hints: tuple[str, ...]


BLOCK_CONTRACTS: dict[str, BlockContract] = {
    "B1": BlockContract(
        block_id="B1",
        required_keys=tuple(sorted(TMI_REQUIRED_KEYS | {"method"})),
        required_dimensions=("seed",),
        file_hints=("base", "tmi_eval_dt_final", "tmi_eval_dt_seed"),
    ),
    "B2": BlockContract(
        block_id="B2",
        required_keys=tuple(
            sorted(
                TMI_REQUIRED_KEYS
                | {"method", "seed", "gold_standard_valid"}
                | set(B4_RETAIN_REQUIRED_FIELDS)
            )
        ),
        required_dimensions=("seed",),
        file_hints=("gold_standard_seed",),
    ),
    "B3": BlockContract(
        block_id="B3",
        required_keys=tuple(
            sorted(
                TMI_REQUIRED_KEYS
                | {"method", "seed", "base_checkpoint"}
                | set(B4_RETAIN_REQUIRED_FIELDS)
            )
        ),
        required_dimensions=("seed",),
        file_hints=("naive_ft_seed",),
    ),
    "B4": BlockContract(
        block_id="B4",
        required_keys=tuple(
            sorted(
                TMI_REQUIRED_KEYS
                | {"method", "seed", "kl_weight", "ascent_steps"}
                | set(B4_RETAIN_REQUIRED_FIELDS)
            )
        ),
        required_dimensions=("seed", "kl_weight", "ascent_steps"),
        file_hints=("ga_refit_",),
    ),
    "B4T": BlockContract(
        block_id="B4T",
        required_keys=tuple(
            sorted(
                TMI_REQUIRED_KEYS
                | {
                    "method",
                    "seed",
                    "alpha",
                    "beta",
                    "stage1_steps",
                    "stage2_steps",
                }
                | set(B4_RETAIN_REQUIRED_FIELDS)
            )
        ),
        required_dimensions=(
            "seed",
            "alpha",
            "beta",
            "stage1_steps",
            "stage2_steps",
        ),
        file_hints=("trajdeleter_",),
    ),
    "B5": BlockContract(
        block_id="B5",
        required_keys=(
            "env",
            "seed",
            "kl_weight",
            "ascent_steps",
            "d4rl_score",
            "forget_auc",
            "forget_auc_ci_low",
            "forget_auc_ci_high",
            "retain_diag_auc",
        ),
        required_dimensions=("seed", "kl_weight", "ascent_steps"),
        file_hints=("ablation/",),
    ),
    "B6": BlockContract(
        block_id="B6",
        required_keys=tuple(sorted(TMI_REQUIRED_KEYS | {"block", "model", "seed"})),
        required_dimensions=("model", "seed"),
        file_hints=("b6_",),
    ),
    "B7": BlockContract(
        block_id="B7",
        required_keys=tuple(
            sorted(TMI_REQUIRED_KEYS | {"block", "model", "forget_ratio", "seed"})
        ),
        required_dimensions=("model", "forget_ratio", "seed"),
        file_hints=("b7_",),
    ),
}


def normalize_ratio(value: float | int | str) -> float:
    return round(float(value), 2)


def expected_blocks_for_env(env_name: str) -> tuple[str, ...]:
    env_key = str(env_name).strip().lower()
    return EXPECTED_BLOCKS_BY_ENV.get(env_key, tuple(BLOCK_CONTRACTS))


def canonical_b4_selection_specs(
    env_name: str,
) -> tuple[tuple[str, Pattern[str]], ...]:
    env_key = str(env_name).strip().lower()
    return ENV_B4_CANONICAL_SELECTION_SPECS.get(
        env_key, DEFAULT_B4_CANONICAL_SELECTION_SPECS
    )


def canonical_b4t_selection_specs(
    env_name: str,
) -> tuple[tuple[str, Pattern[str]], ...]:
    del env_name
    return DEFAULT_B4T_CANONICAL_SELECTION_SPECS


def infer_tmi_method_label(ckpt_path: str) -> str:
    ckpt_str = ckpt_path.lower()
    if "gold_standard" in ckpt_str:
        return "gold_standard"
    if "naive_ft" in ckpt_str:
        return "naive_ft"
    if "trajdeleter" in ckpt_str:
        return "trajdeleter"
    if "ga_refit" in ckpt_str or "unlearning" in ckpt_str:
        return "ga_refit"
    return "base"
