#!/usr/bin/env python3
"""
Visualize what the trained BreathHead predicts on a real audio clip.

Produces a 4-panel PNG:
  1. Waveform
  2. Log-mel spectrogram
  3. Breath probability curve from BreathHead (with threshold line)
  4. Comparison: BreathHead predicted events vs Ruinskiy 2007 baseline events
     (and gold labels if a .breath.json exists next to the WAV)

Usage:
    python visualize_predictions.py \
        --nanopitch /path/to/nanopitch/best.pth \
        --breath-head /path/to/breath_head/best.pth \
        --audio /path/to/clip.wav \
        --output /tmp/prediction.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, write PNG only
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import load_labeled_clip, compute_log_mel, SAMPLE_RATE, HOP_SAMPLES
from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector


def load_breath_head(checkpoint_path: Path) -> BreathHead:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})
    head = BreathHead(
        in_features=384,
        hidden=args_dict.get("hidden", 8),
        kernel_size=5,
    )
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


def run_joint_inference(nanopitch: NanoPitch, head: BreathHead,
                        waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run NanoPitch + BreathHead on one waveform. Returns (vad, pitch, breath_prob)."""
    mel = compute_log_mel(waveform)  # (T, 40)
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, T, 40)

    joint = JointModel(nanopitch, head)
    joint.eval()
    with torch.no_grad():
        vad, pitch, breath = joint(mel_t)
    return (vad.squeeze().numpy(),
            pitch.squeeze().numpy(),
            breath.squeeze().numpy())


def get_predicted_events(breath_prob: np.ndarray, threshold: float = 0.5,
                         min_dur_sec: float = 0.05, hop_sec: float = 0.01) -> list[tuple[float, float]]:
    """Threshold breath probability into (start_sec, end_sec) events."""
    binary = breath_prob >= threshold
    events = []
    in_event = False
    start_frame = 0
    for i, b in enumerate(binary):
        if b and not in_event:
            in_event = True
            start_frame = i
        elif not b and in_event:
            in_event = False
            dur = (i - start_frame) * hop_sec
            if dur >= min_dur_sec:
                events.append((start_frame * hop_sec, i * hop_sec))
    if in_event:
        dur = (len(binary) - start_frame) * hop_sec
        if dur >= min_dur_sec:
            events.append((start_frame * hop_sec, len(binary) * hop_sec))
    return events


def plot(waveform: np.ndarray, mel: np.ndarray, breath_prob: np.ndarray,
         predicted_events: list, ruinskiy_events: list,
         gold_events: list | None, threshold: float,
         title: str, output_path: Path):
    """4-panel diagnostic plot."""
    duration = len(waveform) / SAMPLE_RATE
    t_audio = np.linspace(0, duration, len(waveform))
    t_frame = np.arange(len(breath_prob)) * (HOP_SAMPLES / SAMPLE_RATE)

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2, 1.5, 0.7]})
    fig.suptitle(title, fontsize=11)

    # ── 1. Waveform ──
    axes[0].plot(t_audio, waveform, linewidth=0.4, color="#444")
    axes[0].set_ylabel("Waveform")
    axes[0].set_ylim(-1, 1)
    axes[0].grid(True, alpha=0.2)

    # ── 2. Log-mel spectrogram ──
    axes[1].imshow(mel.T, aspect="auto", origin="lower",
                   extent=[0, duration, 0, mel.shape[1]],
                   cmap="magma", interpolation="nearest")
    axes[1].set_ylabel("Mel band")

    # ── 3. Breath probability curve ──
    axes[2].plot(t_frame, breath_prob, color="#6c8cff", linewidth=1.2,
                 label="BreathHead p(breath)")
    axes[2].axhline(threshold, color="orange", linestyle="--", linewidth=0.8,
                    label=f"threshold = {threshold:.2f}")
    axes[2].fill_between(t_frame, 0, breath_prob,
                         where=breath_prob >= threshold,
                         color="#6c8cff", alpha=0.2)
    axes[2].set_ylabel("p(breath)")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.2)

    # ── 4. Event comparison bars ──
    y_breath = 2
    y_ruinskiy = 1
    y_gold = 0
    for s, e in predicted_events:
        axes[3].axvspan(s, e, ymin=(y_breath + 0.05) / 3,
                        ymax=(y_breath + 0.95) / 3,
                        color="#6c8cff", alpha=0.8)
    for ev in ruinskiy_events:
        axes[3].axvspan(ev.start_sec, ev.end_sec,
                        ymin=(y_ruinskiy + 0.05) / 3,
                        ymax=(y_ruinskiy + 0.95) / 3,
                        color="#ff6c8c", alpha=0.8)
    if gold_events:
        for ev in gold_events:
            axes[3].axvspan(ev["start_sec"], ev["end_sec"],
                            ymin=(y_gold + 0.05) / 3,
                            ymax=(y_gold + 0.95) / 3,
                            color="#4cda7c", alpha=0.8)
    axes[3].set_yticks([0.5, 1.5, 2.5])
    axes[3].set_yticklabels(["Gold (training labels)", "Ruinskiy 2007", "BreathHead"])
    axes[3].set_ylim(0, 3)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_xlim(0, duration)
    axes[3].grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nanopitch", type=Path, required=True,
                   help="NanoPitch checkpoint (frozen backbone)")
    p.add_argument("--breath-head", type=Path, required=True,
                   help="Trained BreathHead checkpoint")
    p.add_argument("--audio", type=Path, required=True,
                   help="WAV file to run inference on")
    p.add_argument("--gold", type=Path, default=None,
                   help="Optional .breath.json with gold/training labels")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=Path, required=True,
                   help="Output PNG path")
    args = p.parse_args()

    print("Loading models...")
    nanopitch = load_backbone_frozen(args.nanopitch, torch.device("cpu"))
    head = load_breath_head(args.breath_head)

    print(f"Loading audio: {args.audio}")
    clip = load_labeled_clip(args.audio, label_path=args.gold)
    waveform = clip.waveform
    duration = len(waveform) / SAMPLE_RATE

    print(f"Running inference on {duration:.1f}s of audio...")
    vad, pitch, breath_prob = run_joint_inference(nanopitch, head, waveform)
    mel = compute_log_mel(waveform)

    print(f"  BreathHead: p_min={breath_prob.min():.3f}, "
          f"p_mean={breath_prob.mean():.3f}, p_max={breath_prob.max():.3f}")
    print(f"  Frames above threshold {args.threshold}: "
          f"{int((breath_prob >= args.threshold).sum())}/{len(breath_prob)} "
          f"({(breath_prob >= args.threshold).mean()*100:.1f}%)")

    predicted_events = get_predicted_events(breath_prob, threshold=args.threshold)
    print(f"  Predicted {len(predicted_events)} breath events")

    print("Running Ruinskiy baseline for comparison...")
    detector = RuinskiyDetector()
    ruinskiy_events = detector.detect_array(waveform.astype(np.float32), SAMPLE_RATE)
    print(f"  Ruinskiy found {len(ruinskiy_events)} events")

    gold_events = clip.breath_events if clip.breath_events else None
    title = (f"{args.audio.name} — BreathHead vs Ruinskiy"
             f"{' vs gold' if gold_events else ''}  ({duration:.1f}s)")
    plot(waveform, mel, breath_prob, predicted_events, ruinskiy_events,
         gold_events, args.threshold, title, args.output)


if __name__ == "__main__":
    main()
