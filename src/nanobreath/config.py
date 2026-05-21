"""Project-wide constants and configurable paths.

The audio framing constants here are hard physical choices (matching the
backbone we attach to); the path constants are conveniences with environment
overrides so the same code runs on different developer machines.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Audio framing (must match the backbone) ─────────────────────────────────
SAMPLE_RATE: int = 16_000      # Hz
HOP_SAMPLES: int = 160         # 10 ms hop
WINDOW_SAMPLES: int = 400      # 25 ms analysis window
N_MELS: int = 40
FFT_SIZE: int = 512
HOP_SEC: float = HOP_SAMPLES / SAMPLE_RATE  # 0.010

# ─── Paths (env-overridable) ─────────────────────────────────────────────────
# Default DATA_DIR is `<repo-root>/data`. We resolve the repo root by walking
# up from this file: src/nanobreath/config.py → src/nanobreath → src → repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("NANOBREATH_DATA_DIR", _REPO_ROOT / "data"))
RUNS_DIR: Path = Path(os.environ.get("NANOBREATH_RUNS_DIR", _REPO_ROOT / "runs"))

# Local backbone location. The NanoPitch backbone (a Smule-confidential
# artifact) is NOT committed to the public repo — it lives under models/nanopitch/
# locally and is gitignored. Drop a compatible `model.py` (defining a NanoPitch
# class) and `best.pth` there to run the project standalone, with no env vars.
_LOCAL_BACKBONE_DIR = _REPO_ROOT / "models" / "nanopitch"

# Path to a NanoPitch-compatible backbone checkpoint.
# Resolution: env var → local models/nanopitch/best.pth → None.
NANOPITCH_CHECKPOINT: Path | None = (
    Path(os.environ["NANOPITCH_CHECKPOINT"])
    if "NANOPITCH_CHECKPOINT" in os.environ
    else ((_LOCAL_BACKBONE_DIR / "best.pth")
          if (_LOCAL_BACKBONE_DIR / "best.pth").exists() else None)
)

# Path to a NanoPitch source tree (for the `from model import NanoPitch` import
# in joint.py).
# Resolution: env var → local models/nanopitch/ → None.
NANOPITCH_SRC_DIR: Path | None = (
    Path(os.environ["NANOPITCH_SRC_DIR"])
    if "NANOPITCH_SRC_DIR" in os.environ
    else (_LOCAL_BACKBONE_DIR
          if (_LOCAL_BACKBONE_DIR / "model.py").exists() else None)
)


def vocalset_dir() -> Path:
    """Conventional location for VocalSet under DATA_DIR."""
    return DATA_DIR / "vocalset"


def labels_dir() -> Path:
    """Conventional location for hand-labeled .breath.json files."""
    return DATA_DIR / "labels"
