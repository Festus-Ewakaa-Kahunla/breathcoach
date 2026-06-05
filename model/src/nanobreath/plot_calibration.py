#!/usr/bin/env python3
"""Reliability diagram + probability histogram for a trained BreathHead checkpoint.

Drops a publication-quality two-panel figure:

  (a) Reliability diagram — predicted vs actual positive rate per bin
  (b) Probability histogram, stratified by ground-truth class

This is the figure that turns "the model is under-confident" into a
quantitative paper claim with an Expected Calibration Error (ECE) number.
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


def collect_val_probs(checkpoint_path: Path, nanopitch_path: Path,
                      label_dir: Path, val_fraction: float = 0.2,
                      seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on the val split and return (probs, labels) per-frame arrays."""
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden = ckpt.get("args", {}).get("hidden", 8)
    head = BreathHead(in_features=384, hidden=hidden).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    nanopitch = load_backbone_frozen(nanopitch_path, device)
    joint = JointModel(nanopitch, head).to(device).eval()

    clips = collect_labeled_clips(label_dir)
    rng = random.Random(seed)
    indices = list(range(len(clips)))
    rng.shuffle(indices)
    val_size = max(1, int(len(clips) * val_fraction))
    val_idx = set(indices[:val_size])
    val_clips = [c for i, c in enumerate(clips) if i in val_idx]

    all_probs, all_labels = [], []
    with torch.no_grad():
        for c in val_clips:
            mel = compute_log_mel(c.waveform).astype(np.float32)
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
            _v, _p, breath = joint(mel_t)
            pred = breath.squeeze().cpu().numpy()
            n = min(len(pred), len(c.breath_labels))
            all_probs.append(pred[:n])
            all_labels.append(c.breath_labels[:n].astype(np.float32))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0, 1, n_bins + 1)
    pred_p, actual_p, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1] if i < n_bins - 1
                                       else probs <= edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            pred_p.append(float("nan"))
            actual_p.append(float("nan"))
        else:
            pred_p.append(float(probs[mask].mean()))
            actual_p.append(float(labels[mask].mean()))
        counts.append(n)
    return edges, np.asarray(pred_p), np.asarray(actual_p), np.asarray(counts)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    _, pred_p, actual_p, counts = reliability_bins(probs, labels, n_bins)
    valid = ~np.isnan(pred_p)
    return float(np.sum(counts[valid] * np.abs(pred_p[valid] - actual_p[valid])) / probs.size)


def plot_calibration(probs: np.ndarray, labels: np.ndarray,
                     output_path: Path, title: str = "Calibration"):
    edges, pred_p, actual_p, counts = reliability_bins(probs, labels)
    e = ece(probs, labels)
    mids = (edges[:-1] + edges[1:]) / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Reliability diagram
    ax1.plot([0, 1], [0, 1], color="#666", linestyle="--", linewidth=1, label="perfect calibration")
    valid = ~np.isnan(pred_p)
    ax1.plot(pred_p[valid], actual_p[valid], "o-", color="#6c8cff", markersize=8,
             linewidth=2, label=f"BreathHead (ECE = {e:.3f})")
    for x, y, n in zip(pred_p[valid], actual_p[valid], counts[valid]):
        ax1.annotate(f"n={n}", (x, y), xytext=(6, 4),
                     textcoords="offset points", fontsize=8, color="#666")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Actual positive rate")
    ax1.set_title("(a) Reliability diagram\n(under-confident = points above diagonal)")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # (b) Probability histogram by class
    ax2.hist(probs[labels < 0.5], bins=np.linspace(0, 1, 41), color="#888",
             alpha=0.6, label="ground-truth: not breath", density=True)
    ax2.hist(probs[labels > 0.5], bins=np.linspace(0, 1, 41), color="#ff6c8c",
             alpha=0.6, label="ground-truth: breath", density=True)
    ax2.axvline(probs.max(), color="red", linestyle="--", linewidth=1,
                label=f"observed max = {probs.max():.3f}")
    ax2.set_xlabel("Predicted breath probability")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) Probability histogram by class")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} (ECE = {e:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--nanopitch", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default="BreathHead calibration")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    print(f"Running inference on val split...")
    probs, labels = collect_val_probs(args.checkpoint, args.nanopitch, args.label_dir,
                                      seed=args.seed)
    print(f"{len(probs)} frames, {int(labels.sum())} positive")
    plot_calibration(probs, labels, args.output, args.title)


if __name__ == "__main__":
    main()
