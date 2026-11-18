"""Compact recurrent actor/critic shared by training and the live duel client.

The action mask is part of the behavior policy.  Sampling and PPO evaluation
must use the *recorded* mask, not one recomputed from a later observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PolicyConfig:
    feature_dim: int
    action_dim: int
    hidden_size: int = 128
    encoder_size: int = 128

    def __post_init__(self) -> None:
        if not (1 <= self.feature_dim <= 4096 and 2 <= self.action_dim <= 1024):
            raise ValueError("invalid observation or action dimensions")
        if not (8 <= self.hidden_size <= 1024 and 8 <= self.encoder_size <= 1024):
            raise ValueError("invalid recurrent policy size")

    def to_dict(self) -> dict:
        return asdict(self)


def masked_distribution(logits: Tensor, mask: Tensor) -> Categorical:
    """Fail closed for empty masks; illegal actions have exactly zero probability."""
    if logits.shape != mask.shape:
        raise ValueError("action mask shape does not match logits")
    mask = mask.to(dtype=torch.bool, device=logits.device)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("each observation must have at least one legal action")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("policy produced non-finite logits")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class RecurrentPolicy(nn.Module):
    """Small observation encoder + GRU + separate policy and value heads.

    Hidden state is [batch, hidden_size].  A reset flag applies before consuming
    the corresponding observation, preventing memory leakage between duels.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.encoder_size),
            nn.LayerNorm(config.encoder_size),
            nn.Tanh(),
            nn.Linear(config.encoder_size, config.encoder_size),
            nn.Tanh(),
        )
        self.gru = nn.GRU(config.encoder_size, config.hidden_size, batch_first=True)
        self.policy_head = nn.Linear(config.hidden_size, config.action_dim)
        self.value_head = nn.Linear(config.hidden_size, 1)
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                # Initialize the three gates independently.
                for block in parameter.chunk(3, 0):
                    nn.init.orthogonal_(block)
            else:
                nn.init.zeros_(parameter)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int = 1, device=None) -> Tensor:
        return torch.zeros(batch_size, self.config.hidden_size, device=device or self.device)

    def forward(
        self,
        features: Tensor,
        hidden: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError("features must have shape [batch, time, feature_dim]")
        if features.shape[1] == 0 or not bool(torch.isfinite(features).all()):
            raise ValueError("empty or non-finite observations")
        batch = features.shape[0]
        if hidden is None:
            hidden = self.initial_state(batch)
        if hidden.shape != (batch, self.config.hidden_size):
            raise ValueError("hidden state shape mismatch")
        encoded = self.encoder(features)
        if episode_starts is None:
            output, final = self.gru(encoded, hidden.unsqueeze(0))
            hidden = final.squeeze(0)
        else:
            if episode_starts.shape != features.shape[:2]:
                raise ValueError("episode_starts must have shape [batch, time]")
            outputs = []
            for time in range(encoded.shape[1]):
                hidden = hidden * (~episode_starts[:, time].bool()).unsqueeze(-1)
                output, final = self.gru(encoded[:, time : time + 1], hidden.unsqueeze(0))
                outputs.append(output)
                hidden = final.squeeze(0)
            output = torch.cat(outputs, dim=1)
        return self.policy_head(output), self.value_head(output).squeeze(-1), hidden

    @torch.no_grad()
    def act(
        self,
        features: Tensor,
        mask: Tensor,
        hidden: Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One observation per batch member, with the same mask semantics as PPO."""
        if features.ndim != 2:
            raise ValueError("act features must have shape [batch, feature_dim]")
        logits, values, hidden = self(features.unsqueeze(1), hidden)
        distribution = masked_distribution(logits[:, 0], mask)
        if deterministic:
            action = distribution.probs.argmax(-1)
        elif generator is not None:
            # Independent CPU sampling makes paired evaluations reproducible
            # without advancing the training or live worker's global RNG.
            action = torch.multinomial(
                distribution.probs.cpu(), 1, generator=generator
            ).squeeze(-1).to(features.device)
        else:
            action = distribution.sample()
        return action, distribution.log_prob(action), values[:, 0], hidden

    def evaluate_actions(
        self,
        features: Tensor,
        masks: Tensor,
        actions: Tensor,
        hidden: Tensor | None = None,
        episode_starts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values, hidden = self(features, hidden, episode_starts)
        distribution = masked_distribution(logits, masks)
        return distribution.log_prob(actions), distribution.entropy(), values, hidden
