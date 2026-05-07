from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")


class MLPPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_layers: Sequence[int] = (256, 256),
        dropout: float = 0.1,
        context_length: int = 1,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.context_length = int(context_length)

        widths = [int(v) for v in hidden_layers if int(v) > 0]
        if not widths:
            widths = [128, 128]

        layers: list[Any] = []
        in_dim = self.state_dim
        for width in widths:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = width

        self.backbone = nn.Sequential(*layers)
        self.predict_action_mean = nn.Linear(in_dim, self.act_dim)
        self.action_log_var = nn.Parameter(torch.zeros(self.act_dim))

        self.apply(self._init_weights)

    def _init_weights(self, module: Any) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        states: Any,
        actions: Any,
        returns_to_go: Any,
        timesteps: Any,
        attention_mask: Any = None,
    ) -> tuple[Any, Any]:
        del actions, returns_to_go, timesteps, attention_mask
        batch_size, seq_len = states.shape[:2]
        x = states.reshape(batch_size * seq_len, self.state_dim)
        x = self.backbone(x)
        action_mean = self.predict_action_mean(x).reshape(
            batch_size, seq_len, self.act_dim
        )
        sigma_sq = torch.clamp(torch.exp(self.action_log_var), min=1e-4)
        return action_mean, sigma_sq

    @torch.no_grad()
    def get_action(
        self,
        states: Any,
        actions: Any,
        returns_to_go: Any,
        timesteps: Any,
    ) -> Any:
        action_mean, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_mean[0, -1]
