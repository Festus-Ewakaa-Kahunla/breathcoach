#!/usr/bin/env python3
"""
Export a binary test fixture for test_breath_head.c.

Loads a trained BreathHead, generates a sequence of random inputs, runs
PyTorch forward, and writes the (weights, inputs, expected_outputs) bundle
to a single binary file that the C test reads.

Binary layout (little-endian throughout):

    "BHTV"               4 bytes magic
    uint32 version       (1)
    uint32 hidden        (8 by default)
    uint32 n_weights     (number of weight floats)
    uint32 n_frames      (e.g., 50)
    float32 * n_weights  weight values (in export_breath_head order)
    float32 * n_frames * 384  input concat features
    float32 * n_frames   expected p(breath) outputs from PyTorch

Usage:
    python export_test_fixture.py path/to/breath_head_best.pth -o fixture.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from nanobreath.deployment.export_breath_head import (
    load_breath_head_checkpoint, extract_breath_head_weights,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=50,
                   help="number of frames in the test sequence")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    head, info = load_breath_head_checkpoint(args.checkpoint)
    kw = info["kwargs"]
    hidden = kw["hidden"]
    weights = extract_breath_head_weights(head)

    # Deterministic random inputs (same seed → reproducible fixture)
    torch.manual_seed(args.seed)
    inputs = torch.randn(1, args.n_frames, 384).float()  # (B=1, T, C=384)

    with torch.no_grad():
        outputs = head(inputs).squeeze().numpy().astype(np.float32)  # (T,)

    inputs_flat = inputs.squeeze().numpy().astype(np.float32).reshape(-1)  # (T*384,)

    print(f"BreathHead: hidden={hidden}  weights={len(weights):,}  "
          f"frames={args.n_frames}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]  "
          f"mean={outputs.mean():.4f}")

    with open(args.output, "wb") as f:
        f.write(b"BHTV")
        f.write(struct.pack("<I", 1))                    # version
        f.write(struct.pack("<I", hidden))
        f.write(struct.pack("<I", len(weights)))
        f.write(struct.pack("<I", args.n_frames))
        f.write(weights.tobytes())
        f.write(inputs_flat.tobytes())
        f.write(outputs.tobytes())

    fsize_kb = args.output.stat().st_size / 1024
    print(f"Wrote fixture: {args.output} ({fsize_kb:.1f} KB)")


if __name__ == "__main__":
    main()
