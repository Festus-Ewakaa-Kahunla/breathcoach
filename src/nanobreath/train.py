#!/usr/bin/env python3
"""
Train the BreathHead on a frozen NanoPitch backbone.

Usage:
    python train.py \
        --nanopitch-checkpoint /path/to/nanopitch/best.pth \
        --label-dir /path/to/labeled/clips \
        --output-dir ./runs/breath_v1 \
        --epochs 100 \
        --hidden 8 \
        --batch-size 8 \
        --seq-len 500

Inputs:
- A NanoPitch checkpoint (frozen backbone)
- A directory of `*.breath.json` label files alongside their `*.wav` audio
  (see prototypes/data/label_format.md for the JSON spec)

Outputs:
- Trained breath-head weights (best by val PR-AUC)
- TensorBoard logs
- Per-epoch eval JSON dumps
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import (
    LabeledClip, collect_labeled_clips, compute_log_mel,
    SAMPLE_RATE, HOP_SAMPLES, N_MELS,
)
from nanobreath.eval import evaluate_clip, FrameMetrics, EventMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Train BreathHead on frozen NanoPitch")
    p.add_argument("--nanopitch-checkpoint", type=Path, required=True,
                   help="path to NanoPitch best.pth (e.g. exp12-mixed-aug)")
    p.add_argument("--label-dir", type=Path, required=True,
                   help="directory containing *.breath.json + matching *.wav")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="where to save weights + logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=500,
                   help="training crop length in frames (10ms each, default 5s)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=8,
                   help="BreathHead conv channel size")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction of clips reserved for validation")
    p.add_argument("--loss", choices=["bce", "focal"], default="focal",
                   help="loss function for breath head")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.25,
                   help="weight for positive class in focal loss")
    p.add_argument("--pos-weight", type=float, default=10.0,
                   help="positive-class weight for BCE (handles ~5% breath imbalance)")
    # Augmentation (training only)
    p.add_argument("--aug-noise-std", type=float, default=0.0,
                   help="std-dev of additive Gaussian noise on log-mel (0 = off, "
                        "0.2 is reasonable since log-mel values span roughly -10..2)")
    p.add_argument("--aug-time-mask", type=int, default=0,
                   help="SpecAugment time mask: max frames to zero out per crop (0 = off)")
    p.add_argument("--aug-freq-mask", type=int, default=0,
                   help="SpecAugment freq mask: max mel bands to zero out per crop (0 = off)")
    p.add_argument("--aug-num-masks", type=int, default=2,
                   help="number of independent time/freq mask draws per crop")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Dataset ───────────────────────────────────────────────────────────────

class BreathDataset(Dataset):
    """Random crops of (mel, breath_labels) from a list of LabeledClips.

    Augmentation (training only — applied per crop):
      - Gaussian noise on log-mel values (aug_noise_std > 0)
      - SpecAugment time-mask: zero a random contiguous time slice (aug_time_mask > 0)
      - SpecAugment freq-mask: zero a random contiguous mel-band slice (aug_freq_mask > 0)
      - Multiple independent draws if aug_num_masks > 1
    """

    def __init__(self, clips: List[LabeledClip], seq_len: int, training: bool = True,
                 aug_noise_std: float = 0.0, aug_time_mask: int = 0,
                 aug_freq_mask: int = 0, aug_num_masks: int = 2):
        self.clips = clips
        self.seq_len = seq_len
        self.training = training
        self.aug_noise_std = aug_noise_std if training else 0.0
        self.aug_time_mask = aug_time_mask if training else 0
        self.aug_freq_mask = aug_freq_mask if training else 0
        self.aug_num_masks = aug_num_masks if training else 0
        # Precompute mel for each clip (small dataset; fits in RAM easily)
        self._mels = [compute_log_mel(c.waveform) for c in clips]
        # Filter out clips that are too short
        self.usable = [
            i for i, m in enumerate(self._mels)
            if m.shape[0] >= seq_len
        ]
        if not self.usable:
            raise ValueError(f"No clips have at least {seq_len} frames "
                             f"({seq_len * 0.01:.1f}s)")

    def __len__(self):
        # Each clip yields ~3 random crops per epoch when training, 1 when not.
        return len(self.usable) * (3 if self.training else 1)

    def _augment(self, mel: np.ndarray) -> np.ndarray:
        """Apply SpecAugment + noise in-place-ish. mel shape (T, 40)."""
        if self.aug_noise_std > 0:
            mel = mel + np.random.randn(*mel.shape).astype(np.float32) * self.aug_noise_std
        T, F = mel.shape
        for _ in range(self.aug_num_masks):
            if self.aug_time_mask > 0:
                w = random.randint(1, max(1, self.aug_time_mask))
                t0 = random.randint(0, max(0, T - w))
                mel[t0:t0 + w, :] = 0.0
            if self.aug_freq_mask > 0:
                w = random.randint(1, max(1, self.aug_freq_mask))
                f0 = random.randint(0, max(0, F - w))
                mel[:, f0:f0 + w] = 0.0
        return mel

    def __getitem__(self, idx):
        clip_idx = self.usable[idx % len(self.usable)]
        mel = self._mels[clip_idx]
        labels = self.clips[clip_idx].breath_labels
        confidence = self.clips[clip_idx].confidence

        n_frames = mel.shape[0]
        if self.training:
            start = random.randint(0, n_frames - self.seq_len)
        else:
            start = 0

        mel_crop = mel[start:start + self.seq_len].astype(np.float32).copy()  # (T, 40)
        lbl_crop = labels[start:start + self.seq_len].astype(np.float32)
        cnf_crop = confidence[start:start + self.seq_len].astype(np.float32)

        if self.training and (self.aug_noise_std > 0 or self.aug_time_mask > 0 or self.aug_freq_mask > 0):
            mel_crop = self._augment(mel_crop)

        return (torch.from_numpy(mel_crop),
                torch.from_numpy(lbl_crop),
                torch.from_numpy(cnf_crop))


# ─── Loss ──────────────────────────────────────────────────────────────────

def focal_loss(probs: torch.Tensor, targets: torch.Tensor,
               alpha: float = 0.25, gamma: float = 2.0,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Sigmoid focal loss as in Lin et al. 2017.

    Args:
        probs: (B, T) sigmoid probabilities
        targets: (B, T) binary {0, 1}
        alpha: positive-class balance term
        gamma: focusing parameter
        weights: (B, T) optional per-frame weights (e.g. confidence)
    """
    eps = 1e-6
    probs = probs.clamp(eps, 1.0 - eps)
    pt = torch.where(targets == 1, probs, 1.0 - probs)
    alpha_t = torch.where(targets == 1, alpha, 1.0 - alpha)
    loss = -alpha_t * (1.0 - pt) ** gamma * torch.log(pt)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def bce_loss(probs: torch.Tensor, targets: torch.Tensor,
             pos_weight: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    eps = 1e-6
    probs = probs.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * targets * torch.log(probs)
             + (1.0 - targets) * torch.log(1.0 - probs))
    if weights is not None:
        loss = loss * weights
    return loss.mean()


# ─── Training ──────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = auto_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print(f"Collecting labeled clips from {args.label_dir}")
    clips = collect_labeled_clips(args.label_dir)
    if len(clips) == 0:
        raise SystemExit(f"No labeled clips found in {args.label_dir}. "
                         f"Did you label any audio yet?")
    total_dur_sec = sum(c.duration_sec for c in clips)
    total_pos_frames = sum(int(c.breath_labels.sum()) for c in clips)
    total_frames = sum(c.n_frames for c in clips)
    print(f"  {len(clips)} clips, {total_dur_sec/60:.1f} min total, "
          f"{total_pos_frames} breath frames "
          f"({100*total_pos_frames/max(1,total_frames):.2f}%)")

    # Train/val split by clip
    rng = random.Random(args.seed)
    indices = list(range(len(clips)))
    rng.shuffle(indices)
    val_size = max(1, int(len(clips) * args.val_fraction))
    val_idx = set(indices[:val_size])
    train_clips = [c for i, c in enumerate(clips) if i not in val_idx]
    val_clips = [c for i, c in enumerate(clips) if i in val_idx]

    train_ds = BreathDataset(
        train_clips, seq_len=args.seq_len, training=True,
        aug_noise_std=args.aug_noise_std,
        aug_time_mask=args.aug_time_mask,
        aug_freq_mask=args.aug_freq_mask,
        aug_num_masks=args.aug_num_masks,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # NOTE: val is run on FULL clips (one at a time, no crop), not via DataLoader.
    # This fixes the bug where 5s val crops sometimes contained zero breath events,
    # making PR-AUC undefined for that epoch.
    val_mels = [compute_log_mel(c.waveform).astype(np.float32) for c in val_clips]
    val_labels_arrs = [c.breath_labels.astype(np.float32) for c in val_clips]
    print(f"  Train: {len(train_clips)} clips, Val: {len(val_clips)} clips "
          f"(val evaluated on FULL clips, "
          f"total {sum(m.shape[0] for m in val_mels)} frames)")

    # ── Model ──
    nanopitch = load_backbone_frozen(args.nanopitch_checkpoint, device)
    breath_head = BreathHead(in_features=384, hidden=args.hidden).to(device)
    print(f"BreathHead params: {breath_head.num_parameters():,}")
    joint = JointModel(nanopitch, breath_head).to(device)

    optimizer = torch.optim.AdamW(breath_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5,
    )

    best_pr_auc = -1.0
    best_path = args.output_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        joint.train()
        breath_head.train()  # only the head trains
        running = 0.0
        n_batches = 0
        for mel, lbl, cnf in train_loader:
            mel = mel.to(device)
            lbl = lbl.to(device)
            cnf = cnf.to(device)

            _vad, _pitch, breath = joint(mel)
            pred = breath.squeeze(-1)  # (B, T)

            if args.loss == "focal":
                loss = focal_loss(pred, lbl, args.focal_alpha, args.focal_gamma, cnf)
            else:
                loss = bce_loss(pred, lbl, args.pos_weight, cnf)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(breath_head.parameters(), 5.0)
            optimizer.step()

            running += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running / max(1, n_batches)

        # ── Validate on FULL clips (not random crops) ──
        breath_head.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for mel, lbls in zip(val_mels, val_labels_arrs):
                mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, T, 40)
                _vad, _pitch, breath = joint(mel_t)
                pred = breath.squeeze().cpu().numpy()  # (T,)
                # Align lengths (compute_log_mel can drop trailing partial frame)
                n = min(len(pred), len(lbls))
                all_probs.append(pred[:n])
                all_labels.append(lbls[:n])
        if all_probs:
            probs_flat = np.concatenate(all_probs)
            labels_flat = np.concatenate(all_labels)
            metrics = evaluate_clip(probs_flat, labels_flat)
            fm: FrameMetrics = metrics["frame"]
            ev100: EventMetrics = metrics["event"][100]
            print(f"Epoch {epoch:3d}  loss={train_loss:.4f}  {fm}  {ev100}")

            if fm.pr_auc > best_pr_auc:
                best_pr_auc = fm.pr_auc
                torch.save({
                    "epoch": epoch,
                    "state_dict": breath_head.state_dict(),
                    "frame_metrics": vars(fm),
                    "event_metrics_100ms": vars(ev100),
                    "args": vars(args),
                }, best_path)
                print(f"  ↑ Saved best to {best_path} (PR-AUC={best_pr_auc:.3f})")
        else:
            print(f"Epoch {epoch:3d}  loss={train_loss:.4f}  (no val data)")

    print(f"\nBest val PR-AUC: {best_pr_auc:.3f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
