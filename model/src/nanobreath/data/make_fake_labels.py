#!/usr/bin/env python3
"""
Generate synthetic .breath.json files for smoke-testing the training pipeline.

WHY THIS EXISTS

Before doing any real hand-labeling, we want to prove the entire pipeline works:
  WAV → load_labeled_clip → BreathDataset → BreathHead → train.py → checkpoint

To do that we need .breath.json files alongside real WAVs. This script makes
fake ones by sprinkling random breath events every ~3-8 seconds.

The fake labels are NOT meaningful — they're just structurally correct JSON so
the training code can be exercised end-to-end. We discard the resulting model
weights; we only care that train.py runs to completion without errors.

USAGE

    python make_fake_labels.py <wav-or-dir> [--out-dir DIR] [--seed N]

    # Generate fake labels for one WAV (writes next to it by default)
    python make_fake_labels.py path/to/f1_dona_vibrato.wav

    # Process all WAVs in a directory (excerpts only — these have natural phrases)
    python make_fake_labels.py path/to/excerpts/ --recursive
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from datetime import date
from pathlib import Path


def wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def generate_fake_events(duration_sec: float, seed: int = 0) -> list[dict]:
    """Generate plausible breath events every 3-8 seconds, 150-300 ms long."""
    rng = random.Random(seed)
    events = []
    t = rng.uniform(1.0, 3.0)  # first breath after a small warm-up
    while t < duration_sec - 0.5:
        dur = rng.uniform(0.15, 0.30)
        events.append({
            "start_sec": round(t, 3),
            "end_sec": round(min(duration_sec - 0.05, t + dur), 3),
            "confidence": "high",
            "source": "synthetic_fake_label_for_smoke_test",
        })
        t += rng.uniform(3.0, 8.0)
    return events


def make_label_for_wav(wav_path: Path, out_dir: Path | None = None,
                       seed: int | None = None) -> Path:
    """Write a .breath.json next to the WAV (or in out_dir if given)."""
    duration = wav_duration_sec(wav_path)
    file_seed = seed if seed is not None else abs(hash(wav_path.stem)) % (2**32)
    events = generate_fake_events(duration, file_seed)

    data = {
        "audio_file": wav_path.name,
        "sample_rate": 16000,
        "duration_sec": round(duration, 4),
        "labeler": "synthetic",
        "label_date": date.today().isoformat(),
        "breath_events": events,
        "notes": "SYNTHETIC FAKE LABELS — for pipeline smoke-test only, NOT for real training.",
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy the WAV path into out_dir too so the loader finds matching audio.
        target_wav = out_dir / wav_path.name
        if not target_wav.exists():
            target_wav.symlink_to(wav_path.resolve())
        json_path = out_dir / wav_path.with_suffix(".breath.json").name
    else:
        json_path = wav_path.with_suffix(".breath.json")

    json_path.write_text(json.dumps(data, indent=2))
    return json_path


def main():
    p = argparse.ArgumentParser(description="Generate fake .breath.json labels for smoke testing")
    p.add_argument("path", type=Path, help="WAV file or directory of WAVs")
    p.add_argument("--out-dir", type=Path, help="write labels here instead of next to WAVs")
    p.add_argument("--seed", type=int, help="explicit seed (default: derived from filename)")
    p.add_argument("--recursive", "-r", action="store_true")
    args = p.parse_args()

    if args.path.is_dir():
        if not args.recursive:
            p.error(f"{args.path} is a directory; pass --recursive to process all WAVs")
        wavs = sorted(args.path.rglob("*.wav"))
        if not wavs:
            print(f"No WAVs under {args.path}", file=sys.stderr)
            sys.exit(1)
        for w in wavs:
            out = make_label_for_wav(w, args.out_dir, args.seed)
            n = len(json.loads(out.read_text())["breath_events"])
            print(f"  ✓ {w.name} → {out.name} ({n} fake events)")
    else:
        out = make_label_for_wav(args.path, args.out_dir, args.seed)
        n = len(json.loads(out.read_text())["breath_events"])
        print(f"Wrote {out} ({n} fake events)")


if __name__ == "__main__":
    main()
