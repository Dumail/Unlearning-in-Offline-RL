"""Train base Decision Transformer on D4RL data."""

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.antmaze_utils import (
    compute_augmented_state_stats,
    default_antmaze_state_mode,
    resolve_antmaze_goal_mode,
    resolve_antmaze_offline_state_mode,
    resolve_antmaze_reward_mode,
)
from src.data_pipeline import load_split_metadata, load_splits, resolve_split_dir
from src.dataset import TrajectoryDataset
from src.model_factory import create_model as model_factory_create_model
from src.trainer import DTTrainer


def create_model(cfg, device):
    model = model_factory_create_model(
        cfg,
        obs_dim=cfg.env.state_dim,
        act_dim=cfg.env.act_dim,
    )
    if not hasattr(model, "to"):
        model_name = str(getattr(cfg, "model", "dt"))
        raise TypeError(
            f"Model '{model_name}' is not yet supported by train_base_dt.py. "
            "This script only supports Gaussian policy models that can be passed directly to DTTrainer."
        )
    return model.to(device)


def _metadata_str(metadata: dict, key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    data_dir = resolve_split_dir(cfg.data_dir, cfg.env.env_name)
    splits, stats = load_splits(data_dir)
    split_metadata = load_split_metadata(data_dir)
    train_trajs = splits["train"]
    print(f"Loaded {len(train_trajs)} training trajectories")

    antmaze_goal_mode = resolve_antmaze_goal_mode(
        str(cfg.env.env_name),
        getattr(cfg, "antmaze_goal_mode", None),
    )
    antmaze_offline_state_mode = resolve_antmaze_offline_state_mode(
        str(cfg.env.env_name),
        getattr(cfg, "antmaze_offline_state_mode", None),
    )
    antmaze_reward_mode = resolve_antmaze_reward_mode(
        str(cfg.env.env_name),
        getattr(cfg, "antmaze_reward_mode", None),
    )
    state_mean, state_std = compute_augmented_state_stats(
        train_trajs,
        antmaze_goal_mode,
        antmaze_offline_state_mode,
    )
    antmaze_eval_backend = str(
        getattr(cfg, "antmaze_eval_backend", "gymnasium_v5")
    ).strip()
    adapter_name = _metadata_str(split_metadata, "adapter_name")
    adapter_fixed_goal = None
    if adapter_name == "antmaze_fixed_goal_v1":
        if antmaze_eval_backend.lower() not in {"d4rl_v2", "d4rl", "v2"}:
            raise ValueError(
                "Detected fixed_goal_v1 AntMaze data adapter directory, but the current evaluation backend is not d4rl_v2. "
                "Please explicitly pass +antmaze_eval_backend=d4rl_v2 to avoid falling back to incorrect Gymnasium Robotics v5 evaluation semantics."
            )
        goal_x = split_metadata.get("adapter_goal_x")
        goal_y = split_metadata.get("adapter_goal_y")
        if goal_x is None or goal_y is None:
            raise ValueError(
                "fixed_goal_v1 metadata is missing adapter_goal_x / adapter_goal_y, unable to set D4RL fixed evaluation goal."
            )
        adapter_fixed_goal = [float(goal_x), float(goal_y)]
    with open_dict(cfg):
        cfg.env.state_dim = int(state_mean.shape[0])
        if not hasattr(cfg, "antmaze_state_mode"):
            cfg.antmaze_state_mode = default_antmaze_state_mode(
                antmaze_goal_mode,
                antmaze_offline_state_mode,
            )
        cfg.antmaze_eval_backend = antmaze_eval_backend
        if adapter_fixed_goal is not None:
            cfg.antmaze_fixed_goal = adapter_fixed_goal

    # Create model
    model = create_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create dataset
    dataset = TrajectoryDataset(
        train_trajs,
        context_length=cfg.train.context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Init wandb
    wandb_run = None
    try:
        import wandb

        os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
        os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(wandb_config, dict):
            wandb_config = {"raw_cfg": wandb_config}
        wandb_config = {str(k): v for k, v in wandb_config.items()}
        wandb_init = getattr(wandb, "init", None)
        if not callable(wandb_init):
            raise RuntimeError("wandb.init is unavailable")
        wandb_run = wandb_init(
            project="unlearning-rl",
            name=f"dt-{cfg.env.env_name}",
            config=wandb_config,
        )
    except Exception as e:
        print(f"wandb init failed: {e}, continuing without logging")

    # Train
    trainer = DTTrainer(model, dataset, cfg, device=device)
    trainer.train(
        wandb_run=wandb_run,
        checkpoint_dir=cfg.checkpoint_dir,
        env_name=cfg.env.env_name,
        target_return=cfg.env.target_return,
        random_score=cfg.env.random_score,
        expert_score=cfg.env.expert_score,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_reward_mode=antmaze_reward_mode,
        antmaze_eval_backend=str(getattr(cfg, "antmaze_eval_backend", "gymnasium_v5")),
        antmaze_fixed_goal=np.asarray(
            getattr(cfg, "antmaze_fixed_goal", None), dtype=np.float32
        )
        if getattr(cfg, "antmaze_fixed_goal", None) is not None
        else None,
    )

    if wandb_run is not None:
        wandb_finish = getattr(wandb_run, "finish", None)
        if callable(wandb_finish):
            wandb_finish()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
