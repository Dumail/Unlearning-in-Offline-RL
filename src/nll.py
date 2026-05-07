"""Gaussian NLL computation and verification for Decision Transformer."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .antmaze_utils import (
    augment_states_with_goal_mode,
    normalize_antmaze_offline_state_mode,
    normalize_antmaze_reward_mode,
    select_antmaze_offline_states,
    transform_antmaze_rewards,
)


def gaussian_nll(
    action_mean: torch.Tensor,  # (B, K, act_dim)
    action_true: torch.Tensor,  # (B, K, act_dim)
    log_var: torch.Tensor,  # (act_dim,)
    mask: torch.Tensor | None = None,  # (B, K)
) -> torch.Tensor:
    """Compute Gaussian NLL loss.

    NLL = 0.5 * (log(sigma^2) + (a - mu)^2 / sigma^2 + log(2*pi))
    Sum over act_dim -> per-token NLL, then masked mean over tokens.
    """
    sigma_sq = torch.clamp(torch.exp(log_var), min=1e-4)

    # Per-dimension NLL: (B, K, act_dim)
    nll = 0.5 * (
        torch.log(sigma_sq)
        + (action_true - action_mean) ** 2 / sigma_sq
        + math.log(2 * math.pi)
    )

    # Sum over action dimensions -> per-token NLL: (B, K)
    nll = nll.sum(dim=-1)

    if mask is not None:
        # Masked mean over valid tokens
        nll = (nll * mask).sum() / mask.sum().clamp(min=1)
    else:
        nll = nll.mean()

    return nll


def compute_trajectory_nll(
    model: nn.Module,
    trajectory: dict,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    context_length: int = 20,
    device: str = "cuda",
    antmaze_goal_mode: str = "none",
    antmaze_offline_state_mode: str = "observations",
    antmaze_reward_mode: str = "none",
) -> float:
    """Compute mean per-token NLL for a single trajectory using sliding window.

    Returns scalar NLL score (for TMI computation).
    """
    model.eval()
    offline_state_mode = normalize_antmaze_offline_state_mode(
        antmaze_offline_state_mode
    )
    reward_mode = normalize_antmaze_reward_mode(antmaze_reward_mode)
    obs = augment_states_with_goal_mode(
        select_antmaze_offline_states(trajectory, offline_state_mode),
        trajectory.get("goals"),
        antmaze_goal_mode,
    )
    acts = trajectory["actions"]
    rewards = transform_antmaze_rewards(trajectory["rewards"], reward_mode)
    traj_len = trajectory["length"]

    # Compute return-to-go
    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        rtg[t] = rewards[t] + rtg[t + 1]

    # Normalize states
    obs_norm = (obs - state_mean) / state_std

    K = context_length
    total_nll = 0.0
    n_tokens = 0

    for t in range(traj_len):
        # Window: [max(0, t-K+1), t+1)
        start = max(0, t - K + 1)
        end = t + 1
        actual_len = end - start

        s = obs_norm[start:end]
        a = acts[start:end]
        r = rtg[start:end]
        timesteps = np.arange(start, end, dtype=np.int64)
        embed_timestep = getattr(model, "embed_timestep", None)
        if embed_timestep is not None and hasattr(embed_timestep, "num_embeddings"):
            timesteps = np.clip(timesteps, 0, int(embed_timestep.num_embeddings) - 1)

        # Pad from left
        pad_len = K - actual_len
        if pad_len > 0:
            s = np.concatenate([np.zeros((pad_len, s.shape[1]), dtype=np.float32), s])
            a = np.concatenate([np.zeros((pad_len, a.shape[1]), dtype=np.float32), a])
            r = np.concatenate([np.zeros(pad_len, dtype=np.float32), r])
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])

        # To tensors
        s_t = torch.from_numpy(s).unsqueeze(0).to(device)
        a_t = torch.from_numpy(a).unsqueeze(0).to(device)
        r_t = torch.from_numpy(r).unsqueeze(0).unsqueeze(-1).to(device)
        ts_t = torch.from_numpy(timesteps).unsqueeze(0).to(device)

        with torch.no_grad():
            action_mean, sigma_sq = model(s_t, a_t, r_t, ts_t)

        # NLL for the last token only
        mu = action_mean[0, -1]  # (act_dim,)
        a_true = torch.from_numpy(acts[t]).to(device)
        log_var = model.action_log_var

        nll_val = 0.5 * (
            torch.log(sigma_sq) + (a_true - mu) ** 2 / sigma_sq + math.log(2 * math.pi)
        )
        total_nll += nll_val.sum().item()
        n_tokens += 1

    return total_nll / max(n_tokens, 1)


def verify_nll(model: nn.Module, device: str = "cuda") -> dict:
    """Run NLL verification suite.

    Checks:
        1. Manual numpy vs torch computation (relative error < 1e-5)
        2. Gradient flow (grad exists for mu and log_var)
        3. Numerical stability (large actions, small sigma, zero diff)
        4. Sigma value report
    """
    results = {}
    act_dim = model.act_dim

    # 1. Manual numpy vs torch
    mu_np = np.random.randn(4, act_dim).astype(np.float32)
    a_np = np.random.randn(4, act_dim).astype(np.float32)
    log_var_np = np.random.randn(act_dim).astype(np.float32) * 0.5

    # Numpy computation
    sigma_sq_np = np.clip(np.exp(log_var_np), 1e-4, None)
    nll_np = 0.5 * (
        np.log(sigma_sq_np) + (a_np - mu_np) ** 2 / sigma_sq_np + np.log(2 * np.pi)
    )
    nll_np_mean = nll_np.sum(axis=-1).mean()

    # Torch computation
    mu_t = torch.from_numpy(mu_np).unsqueeze(0).to(device)  # (1, 4, act_dim)
    a_t = torch.from_numpy(a_np).unsqueeze(0).to(device)
    log_var_t = torch.from_numpy(log_var_np).to(device)

    nll_torch = gaussian_nll(mu_t, a_t, log_var_t).item()

    rel_error = abs(nll_torch - nll_np_mean) / (abs(nll_np_mean) + 1e-8)
    results["numpy_vs_torch_rel_error"] = rel_error
    results["numpy_vs_torch_pass"] = rel_error < 1e-5

    # 2. Gradient flow check
    mu_grad = torch.randn(1, 4, act_dim, device=device, requires_grad=True)
    log_var_grad = torch.randn(act_dim, device=device, requires_grad=True)
    a_target = torch.randn(1, 4, act_dim, device=device)

    loss = gaussian_nll(mu_grad, a_target, log_var_grad)
    loss.backward()

    results["mu_grad_exists"] = (
        mu_grad.grad is not None and mu_grad.grad.abs().sum() > 0
    )
    results["log_var_grad_exists"] = (
        log_var_grad.grad is not None and log_var_grad.grad.abs().sum() > 0
    )

    # 3. Numerical stability
    stability_pass = True
    # Large actions
    mu_large = torch.zeros(1, 1, act_dim, device=device)
    a_large = torch.ones(1, 1, act_dim, device=device) * 100.0
    log_var_zero = torch.zeros(act_dim, device=device)
    nll_large = gaussian_nll(mu_large, a_large, log_var_zero)
    if torch.isnan(nll_large) or torch.isinf(nll_large):
        stability_pass = False

    # Very small sigma
    log_var_small = torch.ones(act_dim, device=device) * (-20.0)
    a_close = torch.ones(1, 1, act_dim, device=device) * 0.01
    nll_small = gaussian_nll(mu_large, a_close, log_var_small)
    if torch.isnan(nll_small) or torch.isinf(nll_small):
        stability_pass = False

    # Zero difference
    a_exact = torch.zeros(1, 1, act_dim, device=device)
    nll_zero = gaussian_nll(mu_large, a_exact, log_var_zero)
    if torch.isnan(nll_zero) or torch.isinf(nll_zero):
        stability_pass = False

    results["numerical_stability_pass"] = stability_pass

    # 4. Sigma values from model
    with torch.no_grad():
        sigma_sq = torch.clamp(torch.exp(model.action_log_var), min=1e-4)
        results["sigma_values"] = sigma_sq.cpu().numpy().tolist()
        results["log_var_values"] = model.action_log_var.cpu().numpy().tolist()

    return results
