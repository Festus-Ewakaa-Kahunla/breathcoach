#!/usr/bin/env python3
"""
Generate pseudo-labels from the Respiro-en breath detector (a *learned* teacher).

WHY

The original weak labels come from the Ruinskiy & Lavner 2007 DSP heuristic,
which over-fires on singing. Respiro-en (Interspeech 2024) is a learned
frame-wise breath detector. It is trained on read *speech* (LibriTTS-R), so it
has a singing domain gap: on ~90% of VocalSet excerpts it fires confidently at a
plausible rate (~5 breaths/min), but on ~10% it produces essentially no
activation (max prob < 0.1). For those clips we fall back to the Ruinskiy
baseline so no clip is left without labels.

This writes the same `.breath.json` format the training pipeline consumes, so it
is a drop-in replacement for generate_pseudo_labels.py — only the teacher differs.

The Respiro-en weights/code are external (MIT, github.com/ydqmkkx/Respiro-en),
not bundled. Point --respiro-dir at a checkout containing modules.py + respiro-en.pt.

USAGE
    python -m nanobreath.data.generate_respiro_labels \
        data/vocalset/FULL/<...>/excerpts \
        --out-dir data/respiro_labels --recursive \
        --respiro-dir teacher/Respiro-en
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np


def _load_respiro(respiro_dir: Path):
    import torch
    sys.path.insert(0, str(respiro_dir))
    from modules import DetectionNet, feature_extractor  # type: ignore
    model = DetectionNet()
    ckpt = torch.load(respiro_dir / "respiro-en.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, feature_extractor


def _respiro_probs(model, feature_extractor, wav: np.ndarray) -> np.ndarray:
    import torch
    feature, length = feature_extractor(wav)
    with torch.no_grad():
        return model(feature, length)[0].cpu().numpy()


def _events_from_probs(probs: np.ndarray, threshold: float, min_frames: int):
    """Contiguous runs above threshold → (start_sec, end_sec) at 10 ms/frame."""
    idx = np.where(probs > threshold)[0]
    if len(idx) == 0:
        return []
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    events = []
    for run in runs:
        if len(run) >= min_frames:
            events.append((run[0] * 0.01, (run[-1] + 1) * 0.01, float(probs[run].max())))
    return events


def _wav_duration_sec(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def label_clip(wav_path: Path, out_dir: Path, model, feature_extractor,
               threshold: float, min_frames: int, fail_prob: float):
    import librosa
    wav, _ = librosa.load(str(wav_path), sr=16000)
    probs = _respiro_probs(model, feature_extractor, wav)
    max_prob = float(probs.max())

    if max_prob < fail_prob:
        # Respiro whiffed on this clip — fall back to the DSP baseline.
        from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector
        det = RuinskiyDetector(correlation_threshold=0.35)
        events = [(e.start_sec, e.end_sec, float(e.score)) for e in det.detect_file(wav_path)]
        source, conf = "ruinskiy_fallback", "low"
    else:
        events = _events_from_probs(probs, threshold, min_frames)
        source, conf = "respiro-en", "high"

    out_dir.mkdir(parents=True, exist_ok=True)
    target_wav = out_dir / wav_path.name
    if not target_wav.exists():
        target_wav.symlink_to(wav_path.resolve())

    data = {
        "audio_file": wav_path.name,
        "sample_rate": 16000,
        "duration_sec": round(_wav_duration_sec(wav_path), 4),
        "labeler": "respiro_en_pseudo",
        "label_date": date.today().isoformat(),
        "breath_events": [
            {"start_sec": round(s, 4), "end_sec": round(e, 4),
             "confidence": conf, "source": source, "score": round(sc, 4)}
            for s, e, sc in events
        ],
        "notes": f"PSEUDO-LABELS from Respiro-en (Interspeech 2024). source={source}, "
                 f"max_prob={max_prob:.3f}, threshold={threshold}.",
    }
    (out_dir / wav_path.with_suffix(".breath.json").name).write_text(json.dumps(data, indent=2))
    return len(events), source


def main():
    p = argparse.ArgumentParser(description="Generate pseudo-labels from Respiro-en")
    p.add_argument("path", type=Path, help="WAV file or directory of WAVs")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--respiro-dir", type=Path, default=Path("teacher/Respiro-en"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-frames", type=int, default=10, help="min breath length (10 ms/frame)")
    p.add_argument("--fail-prob", type=float, default=0.1,
                   help="if Respiro max prob < this, fall back to Ruinskiy")
    p.add_argument("--recursive", "-r", action="store_true")
    args = p.parse_args()

    model, feature_extractor = _load_respiro(args.respiro_dir)

    if args.path.is_dir():
        if not args.recursive:
            p.error(f"{args.path} is a directory; pass --recursive")
        wavs = sorted(args.path.rglob("*.wav"))
    else:
        wavs = [args.path]
    if not wavs:
        print(f"No WAVs under {args.path}", file=sys.stderr)
        sys.exit(1)

    total_events, n_fallback, total_dur = 0, 0, 0.0
    for w in wavs:
        n, source = label_clip(w, args.out_dir, model, feature_extractor,
                               args.threshold, args.min_frames, args.fail_prob)
        dur = _wav_duration_sec(w)
        total_events += n
        total_dur += dur
        n_fallback += (source == "ruinskiy_fallback")
        tag = "  (FALLBACK)" if source == "ruinskiy_fallback" else ""
        print(f"  ✓ {w.name}: {n:2d} events in {dur:5.1f}s ({n/max(dur,1e-6)*60:4.1f}/min){tag}")
    print(f"\nTotal: {total_events} events in {total_dur/60:.1f} min audio "
          f"({total_events/max(total_dur,1e-6)*60:.1f}/min), "
          f"{n_fallback}/{len(wavs)} clips used Ruinskiy fallback")


if __name__ == "__main__":
    main()
