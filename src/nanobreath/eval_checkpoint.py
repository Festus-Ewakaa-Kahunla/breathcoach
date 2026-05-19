#!/usr/bin/env python3
"""
Honest full-clip evaluation of a trained BreathHead checkpoint.

Loads a checkpoint, runs the model on FULL val clips (not crops, not augmented),
reports frame F1, PR-AUC, event F1@{50,100,250}ms.

Usage:
    python eval_checkpoint.py /path/to/best.pth \\
        --nanopitch /path/to/nanopitch/best.pth \\
        --label-dir /tmp/excerpts_filtered \\
        --val-fraction 0.2 --seed 13
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import collect_labeled_clips, compute_log_mel
from nanobreath.eval import evaluate_clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--nanopitch", type=Path, required=True)
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hidden = ckpt.get("args", {}).get("hidden", 8)
    head = BreathHead(in_features=384, hidden=hidden).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    nanopitch = load_backbone_frozen(args.nanopitch, device)
    joint = JointModel(nanopitch, head).to(device).eval()

    print(f"Loading labeled clips from {args.label_dir}...")
    clips = collect_labeled_clips(args.label_dir)
    print(f"  {len(clips)} clips")

    # Same split logic as train.py
    rng = random.Random(args.seed)
    indices = list(range(len(clips)))
    rng.shuffle(indices)
    val_size = max(1, int(len(clips) * args.val_fraction))
    val_idx = set(indices[:val_size])
    val_clips = [c for i, c in enumerate(clips) if i in val_idx]
    print(f"  Val: {len(val_clips)} clips, "
          f"{sum(c.duration_sec for c in val_clips):.1f}s total, "
          f"{sum(int(c.breath_labels.sum()) for c in val_clips)} positive frames")

    all_probs, all_labels = [], []
    with torch.no_grad():
        for c in val_clips:
            mel = compute_log_mel(c.waveform).astype(np.float32)
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
            _vad, _pitch, breath = joint(mel_t)
            pred = breath.squeeze().cpu().numpy()
            n = min(len(pred), len(c.breath_labels))
            all_probs.append(pred[:n])
            all_labels.append(c.breath_labels[:n].astype(np.float32))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    print(f"\nEvaluated on {len(probs)} frames "
          f"({int(labels.sum())} positive, {labels.mean()*100:.2f}% class rate)")
    metrics = evaluate_clip(probs, labels)
    fm = metrics["frame"]
    print(f"\nFrame metrics:")
    print(f"  PR-AUC:        {fm.pr_auc:.4f}")
    print(f"  Best-thresh F1: {fm.f1:.4f}  (P={fm.precision:.4f}, R={fm.recall:.4f}, "
          f"threshold={fm.threshold_at_best_f1:.2f})")
    print(f"\nEvent metrics:")
    for tol_ms in (50, 100, 250):
        em = metrics["event"][tol_ms]
        print(f"  F1@{tol_ms:3d}ms: {em.f1:.4f}  "
              f"(P={em.precision:.4f}, R={em.recall:.4f}, "
              f"{em.n_matched}/{em.n_true_events} matched)")


if __name__ == "__main__":
    main()
