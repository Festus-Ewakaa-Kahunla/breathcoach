"""JointModel — the backbone-coupled wrapper that runs BreathHead jointly with
the upstream pitch tracker.

This module is the only place in the codebase that touches the backbone's
internals (its conv stem, GRU stack, and the existing VAD / pitch heads). The
:class:`BreathHead` itself is portable; this wrapper deliberately is not.

If you want to swap the backbone, this is the file you replace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from nanobreath.config import NANOPITCH_SRC_DIR
from nanobreath.model.breath_head import BreathHead


# ─── Joint forward ───────────────────────────────────────────────────────────

class JointModel(nn.Module):
    """Backbone (frozen) + :class:`BreathHead` (trainable).

    Forward returns ``(vad, pitch, breath)`` for each frame. The backbone is
    expected to expose the internal layers used below — see :func:`forward`
    for the exact attribute names.
    """

    def __init__(self, backbone: nn.Module, breath_head: BreathHead):
        super().__init__()
        self.backbone = backbone
        self.breath_head = breath_head

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run backbone + breath head jointly.

        Args:
            mel: ``(B, T, 40)`` log-mel spectrogram input.

        Returns:
            ``vad``    : ``(B, T, 1)``
            ``pitch``  : ``(B, T, 360)``
            ``breath`` : ``(B, T, 1)``
        """
        bb = self.backbone
        B = mel.size(0)
        device = mel.device

        h1 = torch.zeros(1, B, bb.gru_size, device=device)
        h2 = torch.zeros(1, B, bb.gru_size, device=device)
        h3 = torch.zeros(1, B, bb.gru_size, device=device)

        x = mel.permute(0, 2, 1)
        x = nn.functional.pad(x, (2, 0))
        x = torch.tanh(bb.conv1(x))
        x = nn.functional.pad(x, (2, 0))
        x = torch.tanh(bb.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T, 96)

        g1, _ = bb.gru1(x, h1)
        g2, _ = bb.gru2(g1, h2)
        g3, _ = bb.gru3(g2, h3)

        cat = torch.cat([x, g1, g2, g3], dim=-1)  # (B, T, 384)

        vad = torch.sigmoid(bb.dense_vad(cat))
        pitch = torch.sigmoid(bb.dense_pitch(cat))
        breath = self.breath_head(cat)

        return vad, pitch, breath


# ─── Helpers ─────────────────────────────────────────────────────────────────

def attach_breath_head(backbone: nn.Module, hidden: int = 8) -> JointModel:
    """Wrap an already-loaded frozen backbone with a fresh BreathHead."""
    return JointModel(backbone, BreathHead(in_features=384, hidden=hidden))


def load_backbone_frozen(checkpoint_path: Path,
                         device: torch.device,
                         src_dir: Path | None = None) -> nn.Module:
    """Load a NanoPitch-compatible backbone from a checkpoint, freeze all params.

    The backbone's source tree is not bundled with this repo. Pass its location
    via ``src_dir``, or set the ``NANOPITCH_SRC_DIR`` environment variable.

    Raises:
        RuntimeError: if no source directory is configured.
    """
    src_dir = src_dir or NANOPITCH_SRC_DIR
    if src_dir is None:
        raise RuntimeError(
            "Backbone source directory not configured. Set NANOPITCH_SRC_DIR "
            "in the environment, or pass src_dir= to load_backbone_frozen()."
        )

    src_dir = Path(src_dir)
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Import the upstream model definition. Module name is decided by the source
    # tree; we expect a class named ``NanoPitch`` in a ``model`` module.
    from model import NanoPitch  # type: ignore  # pylint: disable=import-error

    backbone = NanoPitch().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (ckpt.get("state_dict")
                  or ckpt.get("model_state_dict")
                  or ckpt)
    backbone.load_state_dict(state_dict)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


__all__ = ["JointModel", "attach_breath_head", "load_backbone_frozen"]
