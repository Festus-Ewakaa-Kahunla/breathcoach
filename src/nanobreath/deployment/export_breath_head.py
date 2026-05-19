#!/usr/bin/env python3
"""
Export the trained BreathHead weights to JSON for WASM deployment.

Mirrors the pattern of NanoPitch's deployment/export_weights.py — flat float32
array + metadata, ready to be loaded into a C inference engine.

WHY A SEPARATE EXPORTER

The breath head is a small post-hoc head trained on top of frozen NanoPitch
features. We don't want to re-export NanoPitch's weights every time we retrain
the breath head, so the breath head has its own JSON. At inference time the
browser loads BOTH files:
  - model.json       (NanoPitch weights — large, rarely changes)
  - breath_head.json (BreathHead weights — small, may iterate often)

WEIGHT LAYOUT (matches what nanopitch.c will need)

The breath head has three trainable tensors:
  - conv1.weight   shape [hidden, 384, kernel]   (default: [8, 384, 5])
  - conv1.bias     shape [hidden]                (default: [8])
  - conv2.weight   shape [hidden, hidden, kernel] (default: [8, 8, 5])
  - conv2.bias     shape [hidden]                (default: [8])
  - head.weight    shape [1, hidden]              (default: [1, 8])
  - head.bias      shape [1]

Exported in this order, row-major flattened, matching the C convention used by
the existing NanoPitch exporter. Total parameter count for defaults: 15,705.

USAGE

    python export_breath_head.py path/to/best.pth -o breath_head.json
    python export_breath_head.py path/to/best.pth -o breath_head.bin --format binary
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from nanobreath.model.breath_head import BreathHead


def load_breath_head_checkpoint(path: Path) -> tuple[BreathHead, dict]:
    """Load a BreathHead checkpoint saved by train.py.

    Checkpoint format (from train.py save):
        {
            "breath_head_state_dict": OrderedDict,
            "breath_head_kwargs": {"in_features": 384, "hidden": 8, ...},
            "epoch": int, "val_pr_auc": float, ...
        }

    For backward-compat we also accept a raw state_dict.
    """
    warnings.warn(
        "Loading checkpoint via torch.load() executes Python deserialization. "
        "Only export checkpoints from trusted sources.",
        RuntimeWarning,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Three accepted formats:
    #   (a) {"breath_head_state_dict": ..., "breath_head_kwargs": ...}
    #   (b) {"state_dict": ..., "args": {..., "hidden": N}, ...}   ← train.py
    #   (c) raw state_dict (an OrderedDict of tensors)
    if isinstance(ckpt, dict) and "breath_head_state_dict" in ckpt:
        sd = ckpt["breath_head_state_dict"]
        kwargs = ckpt.get("breath_head_kwargs",
                          {"in_features": 384, "hidden": 8, "kernel_size": 5})
        meta = {k: v for k, v in ckpt.items()
                if k not in {"breath_head_state_dict", "breath_head_kwargs"}}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        # Infer shapes from tensors, but prefer values from args when present
        args_dict = ckpt.get("args", {})
        in_features = sd["conv1.weight"].shape[1]
        hidden = args_dict.get("hidden", sd["conv1.weight"].shape[0])
        kernel = sd["conv1.weight"].shape[2]
        kwargs = {"in_features": in_features, "hidden": hidden, "kernel_size": kernel}
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        # Raw state_dict — infer kwargs from tensor shapes
        sd = ckpt
        in_features = sd["conv1.weight"].shape[1]
        hidden = sd["conv1.weight"].shape[0]
        kernel = sd["conv1.weight"].shape[2]
        kwargs = {"in_features": in_features, "hidden": hidden, "kernel_size": kernel}
        meta = {}

    head = BreathHead(**{k: v for k, v in kwargs.items()
                         if k in {"in_features", "hidden", "kernel_size", "dropout"}})
    head.load_state_dict(sd)
    head.eval()
    return head, {"kwargs": kwargs, "meta": meta}


def extract_breath_head_weights(head: BreathHead) -> np.ndarray:
    """Pack BreathHead weights into a flat float32 array in C-loadable order."""
    sd = head.state_dict()
    arrays = [
        sd["conv1.weight"].numpy().flatten(),
        sd["conv1.bias"].numpy().flatten(),
        sd["conv2.weight"].numpy().flatten(),
        sd["conv2.bias"].numpy().flatten(),
        sd["head.weight"].numpy().flatten(),
        sd["head.bias"].numpy().flatten(),
    ]
    return np.concatenate(arrays).astype(np.float32)


def export_json(weights_flat: np.ndarray, info: dict, output_path: Path):
    kw = info["kwargs"]
    data = {
        "format": "breath_head_v1",
        "in_features": kw["in_features"],
        "hidden": kw["hidden"],
        "kernel_size": kw["kernel_size"],
        "n_weights": int(len(weights_flat)),
        # Receptive-field info is useful for the C side to know how much
        # left-padding it needs to apply (causal convolution).
        "left_padding_frames": (kw["kernel_size"] - 1) + (kw["kernel_size"] - 1) * 2,
        "weights": weights_flat.tolist(),
    }
    output_path.write_text(json.dumps(data))
    fsize_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(weights_flat):,} weights → {output_path} ({fsize_kb:.1f} KB)")


def export_binary(weights_flat: np.ndarray, info: dict, output_path: Path):
    """Binary format: same magic-+-header-+-floats pattern as NanoPitch."""
    kw = info["kwargs"]
    with open(output_path, "wb") as f:
        f.write(b"BHWT")  # magic: Breath Head Weights
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", kw["in_features"]))
        f.write(struct.pack("<I", kw["hidden"]))
        f.write(struct.pack("<I", kw["kernel_size"]))
        f.write(struct.pack("<I", len(weights_flat)))
        f.write(weights_flat.tobytes())
    fsize_kb = output_path.stat().st_size / 1024
    print(f"Exported {len(weights_flat):,} weights → {output_path} ({fsize_kb:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description="Export BreathHead weights for WASM")
    p.add_argument("checkpoint", type=Path, help="path to BreathHead .pth")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--format", choices=["json", "binary", "auto"], default="auto")
    args = p.parse_args()

    head, info = load_breath_head_checkpoint(args.checkpoint)
    weights = extract_breath_head_weights(head)

    print(f"BreathHead: in={info['kwargs']['in_features']}  "
          f"hidden={info['kwargs']['hidden']}  "
          f"kernel={info['kwargs']['kernel_size']}")
    print(f"Total weights: {len(weights):,} ({len(weights) * 4 / 1024:.1f} KB float32)")

    fmt = args.format
    if fmt == "auto":
        fmt = "binary" if str(args.output).endswith(".bin") else "json"

    if fmt == "json":
        export_json(weights, info, args.output)
    else:
        export_binary(weights, info, args.output)


if __name__ == "__main__":
    main()
