"""
Run identity utilities for experiment tracking and artifact naming.

Generates unique identifiers for runs based on env, seed, model, and forget_ratio.
"""

from pathlib import Path


class IdentityError(Exception):
    """Raised when required identity components are missing."""

    pass


def get_env_name(cfg) -> str:
    """Extract environment name from config.

    Handles both:
    - cfg.env (string like 'hopper_mr')
    - cfg.env_name (full D4RL name like 'hopper-medium-replay-v2')
    """
    if hasattr(cfg, "env") and isinstance(cfg.env, str):
        env_val = cfg.env
        if "_" in env_val:
            return env_val.split("_")[0]
        return env_val
    elif hasattr(cfg, "env_name") and isinstance(cfg.env_name, str):
        return cfg.env_name.split("-")[0]
    raise IdentityError("Cannot determine env: cfg.env or cfg.env_name not found")


def get_model_name(cfg) -> str:
    """Extract model name from config.

    Expects cfg.model to be set via Hydra config group (e.g., 'dt', 'mlp').
    """
    if not hasattr(cfg, "model"):
        raise IdentityError(
            "Missing required 'model' config. "
            "Please specify model group: 'model=dt' or 'model=mlp'"
        )
    model_val = cfg.model
    if hasattr(model_val, "model_type"):
        return model_val.model_type
    return str(model_val)


def get_forget_ratio(cfg) -> float:
    """Extract forget ratio from config.

    Handles multiple naming conventions:
    - cfg.forget_ratio (float)
    - cfg.unlearn.forget_ratio (if nested)
    - cfg.unlearn.forget_set_ratio (legacy)
    """
    if hasattr(cfg, "forget_ratio"):
        return float(cfg.forget_ratio)
    elif hasattr(cfg, "unlearn"):
        if hasattr(cfg.unlearn, "forget_ratio"):
            return float(cfg.unlearn.forget_ratio)
        if hasattr(cfg.unlearn, "forget_set_ratio"):
            return float(cfg.unlearn.forget_set_ratio)
    raise IdentityError(
        "Cannot determine forget_ratio: "
        "cfg.forget_ratio or cfg.unlearn.forget_ratio not found"
    )


def get_seed(cfg) -> int:
    """Extract seed from config, defaulting to 42."""
    if hasattr(cfg, "seed"):
        return int(cfg.seed)
    return 42


def make_run_identity(cfg) -> str:
    """Generate unique run identity string.

    Format: {env}_{model}_r{seed}_f{ratio}
    Example: hopper_dt_r42_f0.10

    This identity is used for:
    - Checkpoint paths
    - Result directories
    - Artifact naming
    """
    env = get_env_name(cfg)
    model = get_model_name(cfg)
    seed = get_seed(cfg)
    ratio = get_forget_ratio(cfg)

    ratio_str = f"{ratio:.2f}" if ratio < 1 else str(int(ratio))
    return f"{env}_{model}_r{seed}_f{ratio_str}"


def make_run_path(base_dir: Path, cfg, subdir: str = "checkpoints") -> Path:
    """Generate full run path: {base_dir}/{subdir}/{identity}"""
    identity = make_run_identity(cfg)
    return base_dir / subdir / identity
