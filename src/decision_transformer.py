"""Decision Transformer with Gaussian action head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """Standard causal multi-head self-attention."""

    def __init__(
        self, hidden_dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        mask = self.mask
        assert isinstance(mask, Tensor)
        attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attn -> Residual -> LN -> MLP -> Residual."""

    def __init__(
        self, hidden_dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 1024
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer with Gaussian action prediction head.

    Token sequence: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
    Predicts action distribution N(mu, sigma^2) at each step.
    sigma^2 is a global learned parameter (shared across all inputs).
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 1,
        n_layers: int = 3,
        context_length: int = 20,
        dropout: float = 0.1,
        max_ep_len: int = 1000,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        max_tokens = 3 * context_length

        # Embedding layers
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, dropout, max_tokens)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim)

        # Action prediction head (Gaussian)
        self.predict_action_mean = nn.Linear(hidden_dim, act_dim)
        # Global log-variance parameter (shared across all inputs)
        self.action_log_var = nn.Parameter(torch.zeros(act_dim))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,  # (B, K, state_dim)
        actions: torch.Tensor,  # (B, K, act_dim)
        returns_to_go: torch.Tensor,  # (B, K, 1)
        timesteps: torch.Tensor,  # (B, K)
        attention_mask: torch.Tensor | None = None,  # (B, K)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            action_mean: (B, K, act_dim) — predicted mean
            sigma_sq: (act_dim,) — clamped variance (shared)
        """
        B, K = states.shape[:2]

        timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)

        # Embed each modality and add timestep embedding
        time_emb = self.embed_timestep(timesteps)  # (B, K, H)
        state_emb = self.embed_state(states) + time_emb
        action_emb = self.embed_action(actions) + time_emb
        return_emb = self.embed_return(returns_to_go) + time_emb

        # Interleave: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        # Shape: (B, 3*K, H)
        token_emb = torch.stack([return_emb, state_emb, action_emb], dim=2)
        token_emb = token_emb.reshape(B, 3 * K, self.hidden_dim)

        token_emb = self.embed_ln(token_emb)
        token_emb = self.drop(token_emb)

        # Apply transformer blocks
        h = token_emb
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)

        # Extract state token outputs (positions 1, 4, 7, ... = 3*i+1)
        # Action is predicted from state tokens
        state_h = h[:, 1::3, :]  # (B, K, H)

        action_mean = self.predict_action_mean(state_h)  # (B, K, act_dim)
        sigma_sq = torch.clamp(torch.exp(self.action_log_var), min=1e-4)  # (act_dim,)

        return action_mean, sigma_sq

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,  # (1, T, state_dim)
        actions: torch.Tensor,  # (1, T, act_dim)
        returns_to_go: torch.Tensor,  # (1, T, 1)
        timesteps: torch.Tensor,  # (1, T)
    ) -> torch.Tensor:
        """Get deterministic action (mean) for the last timestep.

        Handles context truncation to last K steps.
        """
        K = self.context_length
        T = states.shape[1]

        if T > K:
            states = states[:, -K:]
            actions = actions[:, -K:]
            returns_to_go = returns_to_go[:, -K:]
            timesteps = timesteps[:, -K:]

        action_mean, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_mean[0, -1]  # (act_dim,)
