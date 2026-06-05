#!/usr/bin/env python3
"""Post-hoc Platt scaling for a trained BreathHead checkpoint.

When the model is well-discriminated but badly-calibrated (e.g. v4: PR-AUC 0.66
but max probability 0.35), we can fit a 2-parameter sigmoid recalibration on
the validation logits without retraining:

    p_calibrated = sigmoid(a * logit(p_raw) + b)

`a` and `b` are fit by minimizing binary cross-entropy on the val set. The
parameters are stored alongside the checkpoint and applied at inference.

Usage:
    python -m nanobreath.calibrate fit /path/to/best.pth \
        --nanopitch $NANOPITCH_CHECKPOINT --label-dir data/labels
    # writes a, b into best.pth's 'calibration' field

    python -m nanobreath.calibrate plot /path/to/best.pth \
        --nanopitch $NANOPITCH_CHECKPOINT --label-dir data/labels \
        --output before_after.png

This is a known technique (Platt 1999; Guo et al. 2017 "On Calibration of
Modern Neural Networks") and is exactly the right tool for fixing
under-confident sigmoid heads without retraining.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import collect_labeled_clips, compute_log_mel
from nanobreath.plot_calibration import (
    collect_val_probs, reliability_bins, ece, plot_calibration
)


def fit_platt(probs: np.ndarray, labels: np.ndarray,
              max_iter: int = 200, lr: float = 0.05) -> tuple[float, float]:
    """Fit (a, b) such that sigmoid(a * logit(p) + b) is calibrated on (probs, labels).

    Minimizes binary cross-entropy via Adam. Returns (a, b) as floats.
    """
    eps = 1e-6
    p = torch.from_numpy(np.clip(probs, eps, 1 - eps).astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    z = torch.log(p / (1 - p))  # logits

    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        z_cal = a * z + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z_cal, y)
        loss.backward()
        opt.step()
    return float(a.item()), float(b.item())


def apply_platt(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    z = np.log(p / (1 - p))
    z_cal = a * z + b
    return 1.0 / (1.0 + np.exp(-z_cal))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Fit Platt parameters and store in checkpoint")
    fit.add_argument("checkpoint", type=Path)
    fit.add_argument("--nanopitch", type=Path, required=True)
    fit.add_argument("--label-dir", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=13)

    plot = sub.add_parser("plot", help="Plot before/after calibration with fitted params")
    plot.add_argument("checkpoint", type=Path)
    plot.add_argument("--nanopitch", type=Path, required=True)
    plot.add_argument("--label-dir", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--seed", type=int, default=13)

    args = p.parse_args()
    print("Collecting val probabilities...")
    probs, labels = collect_val_probs(args.checkpoint, args.nanopitch, args.label_dir, seed=args.seed)
    print(f"  {len(probs)} frames, {int(labels.sum())} positive")
    print(f"  raw range: [{probs.min():.4f}, {probs.max():.4f}], "
          f"raw ECE: {ece(probs, labels):.4f}")

    a, b = fit_platt(probs, labels)
    cal = apply_platt(probs, a, b)
    print(f"\nFitted: a = {a:.4f}, b = {b:.4f}")
    print(f"  calibrated range: [{cal.min():.4f}, {cal.max():.4f}]")
    print(f"  calibrated ECE: {ece(cal, labels):.4f}")

    if args.cmd == "fit":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt["calibration"] = {"a": a, "b": b, "type": "platt"}
        torch.save(ckpt, args.checkpoint)
        print(f"\nWrote calibration to {args.checkpoint}")
    else:  # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, p_arr, title in [(ax1, probs, f"Before (ECE = {ece(probs, labels):.3f})"),
                                  (ax2, cal,   f"After Platt scaling (ECE = {ece(cal, labels):.3f})")]:
            edges, pred_p, actual_p, counts = reliability_bins(p_arr, labels)
            ax.plot([0, 1], [0, 1], color="#666", linestyle="--", linewidth=1)
            valid = ~np.isnan(pred_p)
            ax.plot(pred_p[valid], actual_p[valid], "o-", color="#6c8cff", markersize=8, linewidth=2)
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Actual positive rate")
            ax.set_title(title)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Post-hoc Platt scaling: a = {a:.3f}, b = {b:.3f}")
        plt.tight_layout()
        plt.savefig(args.output, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
