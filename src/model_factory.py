from __future__ import annotations

from collections.abc import Mapping
import re
from types import SimpleNamespace
from typing import Any

from .decision_transformer import DecisionTransformer
from .lstm_policy import LSTMPolicy
from .mlp import MLPPolicy


def _read_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_model_name(cfg: Any) -> str:
    model_block = _read_value(cfg, "model")
    model_name = _read_value(cfg, "model_name")

    if isinstance(model_block, str):
        return model_block.strip().lower()

    if model_name is not None:
        return str(model_name).strip().lower()

    if model_block is None:
        return "dt"

    model_type = _read_value(model_block, "model_type")
    if model_type is None:
        return "dt"

    normalized = str(model_type).strip().lower()
    alias_map = {
        "decision_transformer": "dt",
        "dt": "dt",
        "mlp_policy": "mlp",
        "mlp": "mlp",
        "lstm_policy": "lstm",
        "lstm": "lstm",
    }
    return alias_map.get(normalized, normalized)


def infer_dt_model_overrides_from_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, int]:
    embed_state_weight = state_dict.get("embed_state.weight")
    if embed_state_weight is None:
        raise KeyError(
            "state_dict is missing embed_state.weight; cannot infer DT architecture"
        )

    shape = getattr(embed_state_weight, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(
            "embed_state.weight has invalid shape; cannot infer DT architecture"
        )

    hidden_dim = int(shape[0])
    layer_indices: set[int] = set()
    for key in state_dict:
        match = re.match(r"blocks\.(\d+)\.", key)
        if match is not None:
            layer_indices.add(int(match.group(1)))

    if not layer_indices:
        raise KeyError(
            "state_dict is missing transformer blocks; cannot infer DT layer count"
        )

    attn_mask = state_dict.get("blocks.0.attn.mask")
    if attn_mask is None:
        raise KeyError(
            "state_dict is missing blocks.0.attn.mask; cannot infer DT context_length"
        )

    mask_shape = getattr(attn_mask, "shape", None)
    if mask_shape is None or len(mask_shape) != 4:
        raise ValueError(
            "blocks.0.attn.mask has invalid shape; cannot infer DT context_length"
        )

    max_tokens = int(mask_shape[-1])
    if max_tokens % 3 != 0:
        raise ValueError(
            "DT attention mask token count is not a multiple of 3; cannot infer context_length"
        )

    return {
        "embedding_dim": hidden_dim,
        "n_layers": max(layer_indices) + 1,
        "context_length": max_tokens // 3,
    }


def apply_dt_model_overrides(cfg: Any, overrides: Mapping[str, int]) -> dict[str, int]:
    model_cfg = _read_value(cfg, "model")
    train_cfg = _read_value(cfg, "train")
    if model_cfg is None or isinstance(model_cfg, str):
        raise TypeError(
            "cfg.model must be a writable config block to apply DT architecture overrides"
        )
    if train_cfg is None or isinstance(train_cfg, str):
        raise TypeError(
            "cfg.train must be a writable config block to apply DT architecture overrides"
        )

    applied: dict[str, int] = {}
    for key in ("embedding_dim", "n_layers"):
        if key in overrides:
            value = int(overrides[key])
            setattr(model_cfg, key, value)
            applied[key] = value
    if "context_length" in overrides:
        value = int(overrides["context_length"])
        setattr(train_cfg, "context_length", value)
        applied["context_length"] = value
    return applied


def create_model(cfg: Any, obs_dim: int, act_dim: int, **kwargs: Any) -> Any:
    model_name = _normalize_model_name(cfg)

    train_cfg = _read_value(cfg, "train", SimpleNamespace())
    model_cfg = _read_value(cfg, "model", SimpleNamespace())

    hidden_dim = kwargs.get(
        "hidden_dim",
        _read_value(
            model_cfg, "embedding_dim", _read_value(train_cfg, "hidden_dim", 128)
        ),
    )
    n_heads = kwargs.get(
        "n_heads",
        _read_value(model_cfg, "n_heads", _read_value(train_cfg, "n_heads", 1)),
    )
    n_layers = kwargs.get(
        "n_layers",
        _read_value(model_cfg, "n_layers", _read_value(train_cfg, "n_layers", 3)),
    )
    context_length = kwargs.get(
        "context_length", _read_value(train_cfg, "context_length", 20)
    )
    dropout = kwargs.get(
        "dropout",
        _read_value(model_cfg, "dropout", _read_value(train_cfg, "dropout", 0.1)),
    )
    max_ep_len = kwargs.get(
        "max_ep_len",
        _read_value(_read_value(cfg, "env", SimpleNamespace()), "max_ep_len", 1000),
    )

    if model_name == "dt":
        return DecisionTransformer(
            state_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            context_length=context_length,
            dropout=dropout,
            max_ep_len=max_ep_len,
        )

    if model_name == "mlp":
        hidden_layers = kwargs.get(
            "hidden_layers",
            _read_value(model_cfg, "hidden_layers", [hidden_dim, hidden_dim]),
        )
        return MLPPolicy(
            state_dim=obs_dim,
            act_dim=act_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            context_length=context_length,
        )

    if model_name == "lstm":
        return LSTMPolicy(
            state_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            context_length=context_length,
            dropout=dropout,
            max_ep_len=max_ep_len,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Supported models: dt, mlp, lstm. "
        "Use model=dt/model=mlp/model=lstm (or model.model_type=decision_transformer/mlp_policy/lstm_policy)."
    )
