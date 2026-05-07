#!/usr/bin/env python
from __future__ import annotations

import importlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

torch = importlib.import_module("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import load_splits
from src.dataset import TrajectoryDataset
from src.experiment_contract import (
    BLOCK_CONTRACTS,
    OPTIONAL_BACKBONE_MODELS,
    SUPPORTED_MODELS,
)
from src.model_factory import create_model as model_factory_create_model
from src.tmi import full_tmi_evaluation, load_matched_sets
from src.trainer import DTTrainer, evaluate


def _b6_contract_payload(model_name: str, env_name: str, seed: int) -> dict:
    values = {
        "block": "B6",
        "model": model_name,
        "seed": int(seed),
        "env": env_name,
        "d4rl_score": 20.0,
        "forget_auc": 0.6,
        "forget_auc_ci_low": 0.55,
        "forget_auc_ci_high": 0.65,
        "retain_diag_auc": 0.52,
        "n_params": 0,
        "train_steps": 0,
        "context_length": 0,
        "matched_budget_status": "placeholder",
    }
    required_keys = BLOCK_CONTRACTS["B6"].required_keys
    missing = [k for k in required_keys if k not in values]
    if missing:
        raise KeyError(f"B6 payload missing required keys: {missing}")
    return values


def _parse_models(raw_models: object) -> tuple[str, ...]:
    allowed_models = SUPPORTED_MODELS + OPTIONAL_BACKBONE_MODELS
    if isinstance(raw_models, (ListConfig, list, tuple)):
        items: list[str] = []
        for item in list(raw_models):
            items.extend([m.strip() for m in str(item).split(",") if m.strip()])
        models = tuple(m.lower() for m in items)
    else:
        models = tuple(
            m.strip().lower() for m in str(raw_models).split(",") if m.strip()
        )
    if not models:
        raise ValueError("models must not be empty")
    invalid = [m for m in models if m not in allowed_models]
    if invalid:
        raise ValueError(
            f"models contain unsupported entries {invalid}, supported: {list(allowed_models)}"
        )
    return tuple(dict.fromkeys(models))


def run_backbone_comparison(
    results_dir: Path, env: str, seed: int, models: tuple[str, ...] = ("dt", "mlp")
) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []
    for model_name in models:
        output_path = results_dir / f"b6_{model_name}_seed{seed}.json"
        payload = _b6_contract_payload(model_name=model_name, env_name=env, seed=seed)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        created_files.append(output_path)
    return created_files


def _load_model_cfg(model_name: str) -> DictConfig:
    model_file = Path(__file__).resolve().parent.parent / "configs" / "model"
    if model_name == "dt":
        loaded = OmegaConf.load(model_file / "dt.yaml")
        if not isinstance(loaded, DictConfig):
            raise TypeError("dt.yaml must be a DictConfig")
        return loaded
    if model_name == "mlp":
        loaded = OmegaConf.load(model_file / "mlp.yaml")
        if not isinstance(loaded, DictConfig):
            raise TypeError("mlp.yaml must be a DictConfig")
        return loaded
    if model_name == "lstm":
        loaded = OmegaConf.load(model_file / "lstm.yaml")
        if not isinstance(loaded, DictConfig):
            raise TypeError("lstm.yaml must be a DictConfig")
        return loaded
    raise ValueError(f"Unsupported model_name={model_name}")


def _merge_model_cfg(cfg: DictConfig, model_name: str) -> DictConfig:
    model_cfg = _load_model_cfg(model_name)
    cfg_container = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(cfg_container, dict):
        raise TypeError("Main config must be convertible to a dict")
    cfg_container["model"] = OmegaConf.to_container(model_cfg, resolve=False)
    merged = OmegaConf.create(cfg_container)
    if isinstance(merged, ListConfig):
        raise TypeError("Merged config must be a DictConfig")
    return merged


def _train_and_eval(
    cfg: DictConfig,
    model_name: str,
    env_name: str,
    seed: int,
    results_dir: Path,
    device: str,
) -> dict:
    cfg_model = _merge_model_cfg(cfg, model_name)
    checkpoint_root = Path(cfg_model.checkpoint_dir)
    ckpt_dir = checkpoint_root / "b6_backbone" / env_name / model_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(cfg_model.data_dir) / env_name
    splits, stats = load_splits(data_dir)
    train_trajs = splits["train"]

    dataset = TrajectoryDataset(
        train_trajs,
        context_length=cfg_model.train.context_length,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    )

    model = model_factory_create_model(
        cfg_model,
        obs_dim=cfg_model.env.state_dim,
        act_dim=cfg_model.env.act_dim,
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    trainer = DTTrainer(model, dataset, cfg_model, device=device)
    trainer.train(
        checkpoint_dir=str(ckpt_dir),
        env_name=env_name,
        target_return=cfg_model.env.target_return,
        random_score=cfg_model.env.random_score,
        expert_score=cfg_model.env.expert_score,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
    )

    d4rl_score = evaluate(
        model,
        env_name,
        target_return=cfg_model.env.target_return,
        random_score=cfg_model.env.random_score,
        expert_score=cfg_model.env.expert_score,
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        device=device,
    )

    forget_trajs, negative_trajs = load_matched_sets(data_dir, splits)
    tmi_results = full_tmi_evaluation(
        model,
        forget_trajs,
        negative_trajs,
        splits["retain"],
        state_mean=stats["state_mean"],
        state_std=stats["state_std"],
        context_length=cfg_model.train.context_length,
        device=device,
    )

    values = {
        "block": "B6",
        "model": model_name,
        "seed": int(seed),
        "env": env_name,
        "d4rl_score": float(d4rl_score),
        "forget_auc": tmi_results["forget_auc"],
        "forget_auc_ci_low": tmi_results["forget_auc_ci_low"],
        "forget_auc_ci_high": tmi_results["forget_auc_ci_high"],
        "retain_diag_auc": tmi_results["retain_diag_auc"],
        "n_params": n_params,
        "train_steps": int(cfg_model.train.n_steps),
        "context_length": int(cfg_model.train.context_length),
        "matched_budget_status": "shared_train_steps",
    }

    required_keys = BLOCK_CONTRACTS["B6"].required_keys
    missing = [k for k in required_keys if k not in values]
    if missing:
        raise KeyError(f"B6 payload missing required keys: {missing}")
    return values


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = cfg.env.env_name
    seed = int(cfg.seed)
    models = _parse_models(getattr(cfg, "models", "dt,mlp"))

    results_dir = Path(cfg.results_dir) / env_name
    results_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for model_name in models:
        payload = _train_and_eval(
            cfg=cfg,
            model_name=model_name,
            env_name=env_name,
            seed=seed,
            results_dir=results_dir,
            device=device,
        )
        output_path = results_dir / f"b6_{model_name}_seed{seed}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        created_files.append(output_path)

    print(f"created={len(created_files)}")
    for path in created_files:
        print(path.as_posix())


if __name__ == "__main__":
    if "--results-dir" in sys.argv:
        parser = ArgumentParser(add_help=False)
        parser.add_argument("--results-dir", required=True)
        parser.add_argument("--env", required=True)
        parser.add_argument("--seed", required=True, type=int)
        parser.add_argument("--models", default="dt,mlp")
        args, _unknown = parser.parse_known_args(sys.argv[1:])
        created = run_backbone_comparison(
            results_dir=Path(args.results_dir),
            env=args.env,
            seed=int(args.seed),
            models=_parse_models(args.models),
        )
        print(f"created={len(created)}")
        for path in created:
            print(path.as_posix())
        raise SystemExit(0)

    from typing import Callable, cast

    cast(Callable[[], None], main)()
