from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        context_length: int = 20,
        dropout: float = 0.1,
        max_ep_len: int = 1000,
        **_kwargs: object,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_length = int(context_length)
        self.n_layers = int(n_layers)

        # Embedding layers: same modal encoding as DT
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_dim)

        # Input fusion: concatenate four modalities and project to LSTM input dimension
        self.input_proj = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.input_ln = nn.LayerNorm(self.hidden_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=dropout if self.n_layers > 1 else 0.0,
        )

        # Output layer
        self.output_ln = nn.LayerNorm(self.hidden_dim)
        self.predict_action_mean = nn.Linear(self.hidden_dim, self.act_dim)
        self.action_log_var = nn.Parameter(torch.zeros(self.act_dim))

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
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _batch_size, _seq_len = states.shape[:2]
        timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)

        # Embed each modality
        state_emb = self.embed_state(states)
        return_emb = self.embed_return(returns_to_go)
        time_emb = self.embed_timestep(timesteps)

        # Autoregressive: use a_{t-1} to predict a_t, shift actions right by one step
        action_emb = self.embed_action(actions)
        prev_action_emb = torch.zeros_like(action_emb)
        prev_action_emb[:, 1:] = action_emb[:, :-1]

        # Zero out prev_action at padding positions
        if attention_mask is not None:
            prev_action_emb = prev_action_emb * attention_mask.unsqueeze(-1)

        # Fuse four modalities -> LSTM input
        fused = self.input_proj(
            torch.cat([state_emb, return_emb, time_emb, prev_action_emb], dim=-1)
        )
        fused = self.input_ln(fused)

        # Zero out padding positions
        if attention_mask is not None:
            fused = fused * attention_mask.unsqueeze(-1)

        # LSTM forward
        lstm_out, _ = self.lstm(fused)

        # Output
        h = self.output_ln(lstm_out)
        action_mean = self.predict_action_mean(h)
        sigma_sq = torch.clamp(torch.exp(self.action_log_var), min=1e-4)

        return action_mean, sigma_sq

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if states.shape[1] > self.context_length:
            states = states[:, -self.context_length :]
            actions = actions[:, -self.context_length :]
            returns_to_go = returns_to_go[:, -self.context_length :]
            timesteps = timesteps[:, -self.context_length :]

        action_mean, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_mean[0, -1]
