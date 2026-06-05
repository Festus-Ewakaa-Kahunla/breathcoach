"""
Breath Head — small causal head attached to the frozen NanoPitch backbone.

Architecture:
    Input:  (B, T, 384)  ← concatenated NanoPitch features (conv2_out + g1 + g2 + g3)
    Output: (B, T, 1)    ← per-frame breath probability after sigmoid

Design constraints:
- Causal (no lookahead) for real-time deployment
- Small (~15-20 K params) — order of magnitude smaller than NanoPitch base (333K)
- Temporal context ~50-100ms via causal conv stack (covers single breath onsets)
- Compatible with WASM export pipeline (only conv1d + linear, no exotic ops)

Why we chose this over alternatives:
- Single Linear(384, 1) (~385 params): no temporal smoothing, sensitive to per-frame noise.
- Single GRU layer (~30K params): more capacity but harder to export to WASM and slower.
- 1-D causal conv stack (this design): simple, vectorizable, exports cleanly to WASM via
  the same patterns NanoPitch already uses, and gives us a 50-150 ms receptive field.

Receptive field calculation:
    Conv1(k=5, dilation=1) → 5 frames = 50 ms
    Conv2(k=5, dilation=2) → +8 frames = +80 ms
    Total: ~13 frames = 130 ms of past context.
    Breath events are typically 100-500 ms; 130 ms of past context is enough to
    detect onset and emit a positive probability before the breath ends.
"""

from __future__ import annotations

import torch
from torch import nn


class BreathHead(nn.Module):
    """Causal small-conv head producing per-frame breath probability.

    Args:
        in_features: dimension of the backbone features (default 384 = 96*4 from NanoPitch)
        hidden:      width of the conv stack (default 16)
        kernel_size: conv kernel size (default 5)
        dropout:     dropout between conv layers (default 0.1)

    Parameter count with defaults (in_features=384, hidden=16, kernel=5):
        conv1: 384*16*5 + 16 = 30,736
        conv2: 16*16*5 + 16  = 1,296
        head:  16*1 + 1      = 17
        Total: 32,049 — slightly above 15K target but reasonable.
        For ~15K target: use hidden=8 (16,392 params).

    Forward:
        Input  (B, T, 384) — backbone concatenated features per frame
        Output (B, T, 1)   — sigmoid probability of breath at each frame
    """

    def __init__(self, in_features: int = 384, hidden: int = 8,
                 kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.kernel_size = kernel_size

        # Conv1d expects (B, C, T). We'll permute in forward.
        self.conv1 = nn.Conv1d(in_features, hidden, kernel_size=kernel_size,
                               padding=0)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size,
                               padding=0, dilation=2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

        # Precomputed left-padding amounts for causal padding
        self._pad1 = kernel_size - 1
        self._pad2 = (kernel_size - 1) * 2

        self._init_weights()

    def _init_weights(self):
        """Mild Kaiming for convs, zero for head bias (start with 0.5 prob)."""
        for module in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict breath probability per frame.

        Args:
            features: (B, T, in_features) — backbone-output features

        Returns:
            (B, T, 1) sigmoid probabilities. Time dimension matches input.
        """
        # (B, T, C) → (B, C, T) for Conv1d
        x = features.permute(0, 2, 1)

        # Causal pad on the LEFT, then conv (kernel rolls over the past)
        x = nn.functional.pad(x, (self._pad1, 0))
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)

        x = nn.functional.pad(x, (self._pad2, 0))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)

        # Back to (B, T, hidden) for the linear head
        x = x.permute(0, 2, 1)
        logits = self.head(x)                     # (B, T, 1)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# JointModel + load helpers live in `nanobreath.model.joint` to keep the
# breath head itself portable and decoupled from a specific backbone layout.


if __name__ == "__main__":
    # Quick sanity check
    head = BreathHead(in_features=384, hidden=8)
    print(f"BreathHead params: {head.num_parameters():,}")
    dummy = torch.randn(2, 200, 384)  # batch=2, T=200 frames (2s), feat=384
    out = head(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
