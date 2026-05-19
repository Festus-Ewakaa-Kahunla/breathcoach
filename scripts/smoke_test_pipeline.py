#!/usr/bin/env python3
"""End-to-end pipeline smoke test on a single VocalSet clip.

What this verifies (no labels needed, no backbone needed):

1. The WAV loader handles VocalSet's 44.1 kHz files and resamples cleanly.
2. The log-mel spectrogram has the expected shape.
3. The BreathHead runs forward on a dummy 384-dim feature tensor and produces
   sigmoid probabilities in [0, 1].
4. The Ruinskiy & Lavner 2007 baseline runs on real audio and returns events.
5. The PhraseTracker state machine consumes (breath_prob, voiced_prob) streams
   and produces coaching messages.

If this script runs to completion without exceptions, the core pipeline is
wired up correctly and you can start labeling / training.

Usage::

    python scripts/smoke_test_pipeline.py [<wav>]

If no WAV is given, the script picks one from ``data/vocalset/FULL/`` automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector
from nanobreath.config import vocalset_dir, SAMPLE_RATE
from nanobreath.data.dataset import compute_log_mel, load_labeled_clip
from nanobreath.feature.phrase_tracker import PhraseTracker
from nanobreath.model.breath_head import BreathHead


def pick_demo_wav() -> Path:
    """Find a longer-form VocalSet clip suitable for a smoke test."""
    root = vocalset_dir() / "FULL"
    if not root.exists():
        raise SystemExit(
            f"VocalSet not found at {root}.\n"
            "Run `nanobreath-download` first, or pass a WAV path explicitly."
        )
    # Prefer the excerpts subdir (musical phrases with natural breaths)
    candidates = sorted(root.rglob("*/excerpts/**/*.wav"))
    if candidates:
        return candidates[0]
    any_wav = sorted(root.rglob("*.wav"))
    if not any_wav:
        raise SystemExit(f"No WAVs found under {root}")
    return any_wav[0]


def main() -> int:
    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else pick_demo_wav()
    print(f"Smoke-testing on: {wav_path}")

    # ── 1. Audio I/O + mel spectrogram ──
    print("\n[1/5] Loading audio + computing log-mel...")
    clip = load_labeled_clip(wav_path)
    print(f"      duration:    {clip.duration_sec:.2f}s "
          f"({clip.n_frames} frames @ {SAMPLE_RATE} Hz)")
    print(f"      waveform:    shape={clip.waveform.shape}, "
          f"min={clip.waveform.min():.3f}, max={clip.waveform.max():.3f}")
    mel = compute_log_mel(clip.waveform)
    print(f"      log-mel:     shape={mel.shape} (expected (n_frames, 40))")
    assert mel.shape[1] == 40, f"expected 40 mel bands, got {mel.shape[1]}"

    # ── 2. BreathHead forward pass on dummy features ──
    print("\n[2/5] BreathHead forward pass...")
    head = BreathHead(in_features=384, hidden=8)
    n_params = head.num_parameters()
    print(f"      params:      {n_params:,}")
    dummy = torch.randn(1, mel.shape[0], 384)
    with torch.no_grad():
        out = head(dummy)
    print(f"      output:      shape={tuple(out.shape)}, "
          f"range=[{out.min().item():.3f}, {out.max().item():.3f}]")
    assert out.shape == (1, mel.shape[0], 1)
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0

    # ── 3. Ruinskiy & Lavner baseline ──
    print("\n[3/5] Ruinskiy & Lavner 2007 baseline...")
    det = RuinskiyDetector()
    events = det.detect_array(clip.waveform, SAMPLE_RATE)
    print(f"      detected {len(events)} breath events")
    for ev in events[:5]:
        print(f"        {ev.start_sec:6.3f} → {ev.end_sec:6.3f} s  "
              f"({ev.duration_sec * 1000:5.1f} ms, score={ev.score:.3f})")
    if len(events) > 5:
        print(f"        ... and {len(events) - 5} more")

    # ── 4. PhraseTracker on a synthetic breath/voicing stream ──
    print("\n[4/5] PhraseTracker on synthetic stream...")
    tracker = PhraseTracker()
    # Use the Ruinskiy events to drive synthetic breath_prob; treat all
    # non-breath as voiced for the smoke test.
    n = clip.n_frames
    breath_probs = np.full(n, 0.05, dtype=np.float32)
    voiced_probs = np.full(n, 0.9, dtype=np.float32)
    for ev in events:
        start = int(ev.start_sec * 100)  # frames @ 10 ms hop
        end = int(ev.end_sec * 100)
        breath_probs[start:end] = 0.9
        voiced_probs[start:end] = 0.1
    phrase_ends = 0
    for t in range(n):
        out = tracker.step(float(breath_probs[t]), float(voiced_probs[t]))
        if out.phrase_just_ended:
            phrase_ends += 1
    summary = tracker.session_summary()
    print(f"      phrases:     {summary['n_phrases']}")
    print(f"      avg phrase:  {summary['avg_phrase_sec']:.2f}s")
    print(f"      max phrase:  {summary['max_phrase_sec']:.2f}s")

    # ── 5. End-to-end sanity ──
    print("\n[5/5] All pipeline stages ran without exception.")
    print("\nSmoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
