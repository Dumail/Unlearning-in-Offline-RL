"""PyTorch Dataset for Decision Transformer training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .antmaze_utils import (
    augment_states_with_goal_mode,
    normalize_antmaze_offline_state_mode,
    normalize_antmaze_reward_mode,
    select_antmaze_offline_states,
    transform_antmaze_rewards,
)


class TrajectoryDataset(Dataset):
    """Dataset that yields K-length subsequences from trajectories for DT training.

    Each sample contains:
        states (K, state_dim): z-normalized observations
        actions (K, act_dim): raw actions
        returns_to_go (K, 1): discounted return-to-go from each step
        timesteps (K,): absolute timestep indices
        attention_mask (K,): 1 for valid tokens, 0 for padding
    """

    def __init__(
        self,
        trajectories: list[dict],
        context_length: int = 20,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        antmaze_goal_mode: str = "none",
        antmaze_offline_state_mode: str = "observations",
        antmaze_reward_mode: str = "none",
    ):
        self.context_length = context_length
        self.state_mean = state_mean if state_mean is not None else 0.0
        self.state_std = state_std if state_std is not None else 1.0
        self.antmaze_goal_mode = str(antmaze_goal_mode).strip().lower()
        self.antmaze_offline_state_mode = normalize_antmaze_offline_state_mode(
            antmaze_offline_state_mode
        )
        self.antmaze_reward_mode = normalize_antmaze_reward_mode(antmaze_reward_mode)

        self.trajectories = trajectories

        # Precompute return-to-go for each trajectory
        self.rtgs = []
        for traj in trajectories:
            rewards = transform_antmaze_rewards(
                traj["rewards"], self.antmaze_reward_mode
            )
            rtg = np.zeros_like(rewards)
            rtg[-1] = rewards[-1]
            for t in range(len(rewards) - 2, -1, -1):
                rtg[t] = rewards[t] + rtg[t + 1]
            self.rtgs.append(rtg.astype(np.float32))

        # Build flat index: map global idx -> (traj_idx, start_pos)
        # Each trajectory contributes len(traj) samples
        self.index_map = []
        for traj_idx, traj in enumerate(trajectories):
            traj_len = traj["length"]
            for start in range(traj_len):
                self.index_map.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        traj_idx, start_pos = self.index_map[idx]
        traj = self.trajectories[traj_idx]
        K = self.context_length

        obs = augment_states_with_goal_mode(
            select_antmaze_offline_states(traj, self.antmaze_offline_state_mode),
            traj.get("goals"),
            self.antmaze_goal_mode,
        )
        acts = traj["actions"]
        rtg = self.rtgs[traj_idx]
        traj_len = traj["length"]

        state_dim = obs.shape[1]
        act_dim = acts.shape[1]

        # Extract subsequence ending at start_pos (inclusive)
        # We want the window [start_pos - K + 1, start_pos + 1)
        seq_start = max(0, start_pos - K + 1)
        seq_end = start_pos + 1
        actual_len = seq_end - seq_start

        # Extract raw sequences
        s = obs[seq_start:seq_end]
        a = acts[seq_start:seq_end]
        r = rtg[seq_start:seq_end]
        timesteps = np.arange(seq_start, seq_end, dtype=np.int64)

        # Normalize states
        s = (s - self.state_mean) / self.state_std

        # Pad from the left if needed
        pad_len = K - actual_len
        if pad_len > 0:
            s = np.concatenate([np.zeros((pad_len, state_dim), dtype=np.float32), s])
            a = np.concatenate([np.zeros((pad_len, act_dim), dtype=np.float32), a])
            r = np.concatenate([np.zeros(pad_len, dtype=np.float32), r])
            timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), timesteps])
            mask = np.concatenate(
                [
                    np.zeros(pad_len, dtype=np.float32),
                    np.ones(actual_len, dtype=np.float32),
                ]
            )
        else:
            mask = np.ones(K, dtype=np.float32)

        return {
            "states": torch.from_numpy(s),
            "actions": torch.from_numpy(a),
            "returns_to_go": torch.from_numpy(r).unsqueeze(-1),
            "timesteps": torch.from_numpy(timesteps),
            "attention_mask": torch.from_numpy(mask),
        }
