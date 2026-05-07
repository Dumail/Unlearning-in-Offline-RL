"""Training loop for Decision Transformer."""

from __future__ import annotations

import contextlib
import io
import importlib
import math
import os
from pathlib import Path
from typing import Any, Sequence, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .antmaze_utils import augment_states_with_goal_mode, normalize_antmaze_reward_mode
from .dataset import TrajectoryDataset
from .decision_transformer import DecisionTransformer
from .env_utils import (
    extract_state_for_env_name,
    extract_success_flag,
    get_env_spec,
    score_rollouts,
)
from .nll import gaussian_nll


def _build_eval_rtg_sequence(
    target_return: float, rewards_hist: Sequence[float], start: int
) -> np.ndarray:
    prefix_rewards = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(rewards_hist, dtype=np.float32)]
    )
    rtg = np.float32(target_return) - prefix_rewards[start:]
    return rtg.reshape(-1, 1)


def _transform_eval_reward(reward: float, antmaze_reward_mode: str) -> float:
    mode = normalize_antmaze_reward_mode(antmaze_reward_mode)
    if mode == "minus_one":
        return float(reward) - 1.0
    return float(reward)


def _normalize_antmaze_eval_backend(backend: str | None) -> str:
    mode = str(backend or "gymnasium_v5").strip().lower()
    if mode in {"gymnasium_v5", "gymnasium", "v5"}:
        return "gymnasium_v5"
    if mode in {"d4rl_v2", "d4rl", "v2"}:
        return "d4rl_v2"
    raise ValueError(
        f"Unknown AntMaze evaluation backend: {backend}. Valid options: gymnasium_v5, d4rl_v2"
    )


def _unwrap_d4rl_antmaze(env: Any) -> Any:
    base_env = env.unwrapped
    if hasattr(base_env, "wrapped_env"):
        base_env = base_env.wrapped_env
    if not hasattr(base_env, "set_state") or not hasattr(base_env, "set_target_goal"):
        raise TypeError(f"Not an expected D4RL AntMaze environment: {type(base_env)}")
    return base_env


def _extract_d4rl_qpos_qvel_state(env: Any) -> np.ndarray:
    base_env = _unwrap_d4rl_antmaze(env)
    qpos = np.asarray(base_env.data.qpos, dtype=np.float32)
    qvel = np.asarray(base_env.data.qvel, dtype=np.float32)
    return np.concatenate([qpos, qvel], axis=0)


def _extract_d4rl_target_goal(env: Any) -> np.ndarray:
    base_env = _unwrap_d4rl_antmaze(env)
    target_goal = getattr(base_env, "target_goal", None)
    if target_goal is None:
        raise AttributeError("D4RL AntMaze environment is missing target_goal")
    return np.asarray(target_goal, dtype=np.float32)


def _set_d4rl_target_goal(env: Any, goal: np.ndarray) -> None:
    base_env = _unwrap_d4rl_antmaze(env)
    with contextlib.redirect_stdout(io.StringIO()):
        base_env.set_target_goal(tuple(np.asarray(goal, dtype=np.float64).tolist()))


def _seed_eval_env(env: Any, eval_seed: int | None) -> None:
    if eval_seed is None:
        return
    if hasattr(env, "seed"):
        env.seed(int(eval_seed))


def _extract_antmaze_state_for_d4rl_env(
    env: Any,
    obs: np.ndarray,
    antmaze_state_mode: str,
    prev_qpos_qvel_state: np.ndarray | None = None,
) -> np.ndarray:
    mode = str(antmaze_state_mode).strip().lower()
    observation = np.asarray(obs, dtype=np.float32)
    qpos_qvel_state = _extract_d4rl_qpos_qvel_state(env)
    goal = _extract_d4rl_target_goal(env)

    if mode == "qpos_qvel":
        return qpos_qvel_state
    if mode == "prev_qpos_qvel":
        if prev_qpos_qvel_state is not None:
            return np.asarray(prev_qpos_qvel_state, dtype=np.float32)
        return qpos_qvel_state
    if mode == "observation_first29":
        return observation[:29]
    if mode == "qpos_qvel_goal":
        return augment_states_with_goal_mode(
            qpos_qvel_state[None, :], goal[None, :], "append_goal"
        )[0]
    if mode == "prev_qpos_qvel_goal":
        base_state = (
            np.asarray(prev_qpos_qvel_state, dtype=np.float32)
            if prev_qpos_qvel_state is not None
            else qpos_qvel_state
        )
        return augment_states_with_goal_mode(
            base_state[None, :], goal[None, :], "append_goal"
        )[0]
    if mode == "observation_first29_goal":
        return augment_states_with_goal_mode(
            observation[:29][None, :], goal[None, :], "append_goal"
        )[0]
    if mode == "qpos_qvel_relative_goal":
        return augment_states_with_goal_mode(
            qpos_qvel_state[None, :], goal[None, :], "relative_goal"
        )[0]
    if mode == "prev_qpos_qvel_relative_goal":
        base_state = (
            np.asarray(prev_qpos_qvel_state, dtype=np.float32)
            if prev_qpos_qvel_state is not None
            else qpos_qvel_state
        )
        return augment_states_with_goal_mode(
            base_state[None, :], goal[None, :], "relative_goal"
        )[0]
    if mode == "observation_first29_relative_goal":
        return augment_states_with_goal_mode(
            observation[:29][None, :], goal[None, :], "relative_goal"
        )[0]
    raise ValueError(
        "Unknown AntMaze state extraction mode: "
        f"{antmaze_state_mode}. Valid options: qpos_qvel, prev_qpos_qvel, observation_first29, qpos_qvel_goal, prev_qpos_qvel_goal, observation_first29_goal, qpos_qvel_relative_goal, prev_qpos_qvel_relative_goal, observation_first29_relative_goal"
    )


def _evaluate_antmaze_d4rl(
    model: DecisionTransformer,
    env_name: str,
    n_episodes: int,
    target_return: float,
    random_score: float,
    expert_score: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: str,
    max_ep_len: int,
    antmaze_state_mode: str,
    antmaze_reward_mode: str,
    antmaze_fixed_goal: np.ndarray | None,
    eval_seed: int | None,
) -> float:
    importlib.import_module("d4rl")
    gym = importlib.import_module("gym")

    env = gym.make(env_name)
    _seed_eval_env(env, eval_seed)
    action_space = cast(Any, env.action_space)

    returns = []
    successes = []
    for episode_idx in range(n_episodes):
        episode_seed = int(episode_idx)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                reset_out = env.reset(seed=episode_seed)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(episode_seed)
                reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        if antmaze_fixed_goal is not None:
            _set_d4rl_target_goal(env, antmaze_fixed_goal)
        prev_qpos_qvel_state = None
        state = _extract_antmaze_state_for_d4rl_env(
            env,
            obs,
            antmaze_state_mode=antmaze_state_mode,
            prev_qpos_qvel_state=prev_qpos_qvel_state,
        )

        states_hist = []
        actions_hist = []
        rewards_hist = []
        episode_return = 0.0
        episode_success = False
        target_rtg = target_return

        for _t in range(max_ep_len):
            states_hist.append(state)
            K = model.context_length
            hist_len = len(states_hist)
            start = max(0, hist_len - K)

            s = np.array(
                [
                    (np.array(states_hist[i], dtype=np.float32) - state_mean)
                    / state_std
                    for i in range(start, hist_len)
                ]
            )
            if len(actions_hist) > 0:
                a = np.array(actions_hist[start:hist_len], dtype=np.float32)
                if len(a) < len(s):
                    a = np.concatenate(
                        [a, np.zeros((1, model.act_dim), dtype=np.float32)]
                    )
            else:
                a = np.zeros((len(s), model.act_dim), dtype=np.float32)

            rtg = _build_eval_rtg_sequence(target_rtg, rewards_hist, start)
            timesteps = np.arange(start, start + len(s), dtype=np.int64)
            max_timestep = 999
            embed_timestep = getattr(model, "embed_timestep", None)
            if embed_timestep is not None and hasattr(embed_timestep, "num_embeddings"):
                max_timestep = int(embed_timestep.num_embeddings) - 1
            timesteps = np.clip(timesteps, 0, max_timestep)

            s_t = torch.from_numpy(s).unsqueeze(0).float().to(device)
            a_t = torch.from_numpy(a).unsqueeze(0).float().to(device)
            rtg_t = torch.from_numpy(rtg).unsqueeze(0).float().to(device)
            ts_t = torch.from_numpy(timesteps).unsqueeze(0).long().to(device)

            action = model.get_action(s_t, a_t, rtg_t, ts_t).cpu().numpy()
            action = np.clip(action, action_space.low, action_space.high)

            obs, reward, done, info = env.step(action)
            transformed_reward = _transform_eval_reward(
                float(reward), antmaze_reward_mode
            )
            current_qpos_qvel_state = _extract_d4rl_qpos_qvel_state(env)
            state = _extract_antmaze_state_for_d4rl_env(
                env,
                obs,
                antmaze_state_mode=antmaze_state_mode,
                prev_qpos_qvel_state=prev_qpos_qvel_state,
            )
            prev_qpos_qvel_state = current_qpos_qvel_state
            episode_return += float(reward)
            episode_success = (
                episode_success or float(reward) > 0.0 or extract_success_flag(info)
            )
            actions_hist.append(action)
            rewards_hist.append(transformed_reward)
            if done:
                break

        returns.append(episode_return)
        successes.append(episode_success)

    env.close()
    return score_rollouts(env_name, returns, random_score, expert_score, successes)


class DTTrainer:
    """Trainer for Decision Transformer with Gaussian NLL loss."""

    def __init__(
        self,
        model: DecisionTransformer,
        train_dataset: TrajectoryDataset,
        cfg,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.train_iter = iter(self.train_loader)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        self.n_steps = cfg.train.n_steps
        self.warmup_steps = cfg.train.warmup_steps
        self.grad_clip = cfg.train.grad_clip

    def get_lr(self, step: int) -> float:
        """Linear warmup then cosine decay."""
        if step < self.warmup_steps:
            return self.cfg.train.lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.n_steps - self.warmup_steps)
        return self.cfg.train.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_batch(self) -> dict:
        """Get next batch, cycling through data."""
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_step(self, step: int) -> dict:
        """Single training step. Returns metrics dict."""
        self.model.train()

        # Update learning rate
        lr = self.get_lr(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        batch = self.get_batch()
        action_mean, sigma_sq = self.model(
            batch["states"],
            batch["actions"],
            batch["returns_to_go"],
            batch["timesteps"],
            batch["attention_mask"],
        )

        loss = gaussian_nll(
            action_mean,
            batch["actions"],
            self.model.action_log_var,
            mask=batch["attention_mask"],
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "lr": lr,
            "sigma_mean": sigma_sq.mean().item(),
        }

    def train(
        self,
        wandb_run=None,
        checkpoint_dir: str | Path = "checkpoints",
        env_name: str | None = None,
        target_return: float = 6000.0,
        random_score: float = 0.0,
        expert_score: float = 1.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        antmaze_reward_mode: str = "none",
        antmaze_eval_backend: str = "gymnasium_v5",
        antmaze_fixed_goal: np.ndarray | None = None,
    ) -> None:
        """Full training loop."""
        checkpoint_dir_path = Path(checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        log_interval = max(1, self.n_steps // 10)
        metrics: dict[str, float] = {"loss": float("nan")}
        for step in range(self.n_steps):
            metrics = self.train_step(step)

            if step % log_interval == 0 or step == self.n_steps - 1:
                pct = step / self.n_steps * 100
                print(
                    f"[{pct:5.1f}%] step {step}/{self.n_steps}  "
                    f"loss={metrics['loss']:.4f}  lr={metrics['lr']:.6f}  "
                    f"sigma={metrics['sigma_mean']:.4f}"
                )

            if wandb_run is not None and (
                step % log_interval == 0 or step == self.n_steps - 1
            ):
                wandb_run.log(metrics, step=step)

            # Evaluation
            if env_name and step > 0 and step % self.cfg.train.eval_interval == 0:
                eval_episodes = int(
                    getattr(
                        self.cfg.train,
                        "eval_episodes",
                        getattr(self.cfg, "eval_episodes", 10),
                    )
                )
                max_ep_len = int(getattr(self.cfg.env, "max_ep_len", 1000))
                antmaze_state_mode = str(
                    getattr(self.cfg, "antmaze_state_mode", "qpos_qvel")
                )
                eval_score = evaluate(
                    self.model,
                    env_name,
                    n_episodes=eval_episodes,
                    target_return=target_return,
                    random_score=random_score,
                    expert_score=expert_score,
                    state_mean=state_mean,
                    state_std=state_std,
                    antmaze_reward_mode=antmaze_reward_mode,
                    device=self.device,
                    max_ep_len=max_ep_len,
                    antmaze_state_mode=antmaze_state_mode,
                    antmaze_eval_backend=antmaze_eval_backend,
                    antmaze_fixed_goal=antmaze_fixed_goal,
                )
                print(f"\nStep {step}: D4RL score = {eval_score:.2f}")
                if wandb_run is not None:
                    wandb_run.log({"eval/d4rl_score": eval_score}, step=step)

            # Checkpointing
            if step > 0 and step % self.cfg.train.save_interval == 0:
                ckpt_path = checkpoint_dir_path / f"dt_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": metrics["loss"],
                    },
                    ckpt_path,
                )
                print(f"\nSaved checkpoint: {ckpt_path}")

        # Final checkpoint
        ckpt_path = checkpoint_dir_path / "dt_final.pt"
        torch.save(
            {
                "step": self.n_steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": metrics["loss"],
            },
            ckpt_path,
        )
        print(f"Training complete. Final checkpoint: {ckpt_path}")


def evaluate(
    model: DecisionTransformer,
    env_name: str,
    n_episodes: int = 10,
    target_return: float = 6000.0,
    random_score: float = 0.0,
    expert_score: float = 1.0,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    device: str = "cuda",
    max_ep_len: int = 1000,
    antmaze_state_mode: str = "qpos_qvel",
    antmaze_reward_mode: str = "none",
    antmaze_eval_backend: str = "gymnasium_v5",
    antmaze_fixed_goal: np.ndarray | None = None,
    eval_seed: int | None = None,
) -> float:
    """Evaluate DT in gymnasium env with greedy (mean) actions.

    Returns D4RL normalized score: 100 * (ret - random) / (expert - random)
    """
    model.eval()
    state_mean = state_mean if state_mean is not None else np.zeros(model.state_dim)
    state_std = state_std if state_std is not None else np.ones(model.state_dim)

    backend = _normalize_antmaze_eval_backend(antmaze_eval_backend)
    if env_name.startswith("antmaze-") and backend == "d4rl_v2":
        return _evaluate_antmaze_d4rl(
            model,
            env_name,
            n_episodes,
            target_return,
            random_score,
            expert_score,
            state_mean,
            state_std,
            device,
            max_ep_len,
            antmaze_state_mode,
            antmaze_reward_mode,
            antmaze_fixed_goal,
            eval_seed,
        )

    spec = get_env_spec(env_name)
    env = gym.make(spec.gym_name)
    action_space = cast(Any, env.action_space)

    returns = []
    successes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        prev_qpos_qvel_state = None
        state = extract_state_for_env_name(
            env_name,
            env,
            obs,
            antmaze_state_mode=antmaze_state_mode,
            prev_qpos_qvel_state=prev_qpos_qvel_state,
        )
        done = False

        # Histories for context
        states_hist = []
        actions_hist = []
        rewards_hist = []

        episode_return = 0.0
        episode_success = False
        target_rtg = target_return

        for t in range(max_ep_len):
            states_hist.append(state)

            # Build tensors from history
            K = model.context_length
            hist_len = len(states_hist)
            start = max(0, hist_len - K)

            s = np.array(
                [
                    (np.array(states_hist[i], dtype=np.float32) - state_mean)
                    / state_std
                    for i in range(start, hist_len)
                ]
            )
            if len(actions_hist) > 0:
                a = np.array(actions_hist[start:hist_len], dtype=np.float32)
                # Pad actions to same length as states (actions are one behind)
                if len(a) < len(s):
                    a = np.concatenate(
                        [a, np.zeros((1, model.act_dim), dtype=np.float32)]
                    )
            else:
                a = np.zeros((len(s), model.act_dim), dtype=np.float32)

            rtg = _build_eval_rtg_sequence(target_rtg, rewards_hist, start)

            timesteps = np.arange(start, start + len(s), dtype=np.int64)
            max_timestep = 999
            embed_timestep = getattr(model, "embed_timestep", None)
            if embed_timestep is not None and hasattr(embed_timestep, "num_embeddings"):
                max_timestep = int(embed_timestep.num_embeddings) - 1
            timesteps = np.clip(timesteps, 0, max_timestep)

            s_t = torch.from_numpy(s).unsqueeze(0).float().to(device)
            a_t = torch.from_numpy(a).unsqueeze(0).float().to(device)
            rtg_t = torch.from_numpy(rtg).unsqueeze(0).float().to(device)
            ts_t = torch.from_numpy(timesteps).unsqueeze(0).long().to(device)

            action = model.get_action(s_t, a_t, rtg_t, ts_t)
            action = action.cpu().numpy()

            # Clip actions to env bounds
            action = np.clip(action, action_space.low, action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            transformed_reward = _transform_eval_reward(
                float(reward), antmaze_reward_mode
            )
            current_qpos_qvel_state = None
            if env_name.startswith("antmaze-"):
                current_qpos_qvel_state = extract_state_for_env_name(
                    env_name,
                    env,
                    obs,
                    antmaze_state_mode="qpos_qvel",
                )
            state = extract_state_for_env_name(
                env_name,
                env,
                obs,
                antmaze_state_mode=antmaze_state_mode,
                prev_qpos_qvel_state=prev_qpos_qvel_state,
            )
            prev_qpos_qvel_state = current_qpos_qvel_state
            done = terminated or truncated
            episode_return += float(reward)
            episode_success = episode_success or extract_success_flag(info)
            actions_hist.append(action)
            rewards_hist.append(transformed_reward)

            if done:
                break

        returns.append(episode_return)
        successes.append(episode_success)

    env.close()

    d4rl_score = score_rollouts(
        env_name, returns, random_score, expert_score, successes
    )
    return d4rl_score
