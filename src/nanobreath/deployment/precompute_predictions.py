#!/usr/bin/env python3
"""
Precompute breath predictions for one or more WAV files → JSON + spectrogram PNG.

WHY

The browser demo plays an audio file and visualizes breath events + phrase length
in sync. Until we have WASM inference live in the browser, we precompute
everything in Python here and ship the result as a single JSON the browser loads.

OUTPUT (for each input WAV "foo.wav"):
  - foo.json    — per-frame predictions + events (see schema below)
  - foo.png     — log-mel spectrogram pre-rendered, viridis colormap
  - foo.wav     — symlink to source audio (so the browser fetch finds it)

SCHEMA (foo.json):
    {
      "audio_file": "foo.wav",
      "spectrogram_file": "foo.png",
      "duration_sec": 21.00,
      "sample_rate": 16000,
      "hop_sec": 0.01,
      "n_frames": 2100,
      "breath_prob": [0.012, ...],
      "voiced_prob": [0.001, ...],
      "pitch_norm":  [0.34, ...],
      "predicted_events": [{"start_sec":4.32, "end_sec":4.61, "score":0.78}, ...],
      "ruinskiy_events":  [...],
      "phrase_events":    [{"start_sec":0.30, "end_sec":4.32, "duration_sec":4.02}, ...],
      "threshold": 0.15,
      "model_meta": {"hidden": 8, "params": 15705, "inference_ms": 24.5}
    }

USAGE
    # Single file (writes to --output-dir/foo.{json,png,wav})
    python precompute_predictions.py \
        --nanopitch path/to/nanopitch.pth \
        --breath-head path/to/breath_head.pth \
        --audio path/to/clip.wav \
        --output-dir web/clips/

    # Batch (process every WAV in --audio-dir)
    python precompute_predictions.py \
        --nanopitch path/to/nanopitch.pth \
        --breath-head path/to/breath_head.pth \
        --audio-dir path/to/clips/ \
        --output-dir web/clips/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import load_labeled_clip, compute_log_mel, SAMPLE_RATE, HOP_SAMPLES
from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector


HOP_SEC = HOP_SAMPLES / SAMPLE_RATE


def load_nanopitch(ckpt_path: Path):
    """Thin convenience wrapper around load_backbone_frozen for CPU."""
    return load_backbone_frozen(ckpt_path, torch.device("cpu"))


def load_breath_head(ckpt_path: Path) -> tuple[BreathHead, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt.get("args", {})
    hidden = args_dict.get("hidden", 8)
    head = BreathHead(in_features=384, hidden=hidden, kernel_size=5)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    # Store calibration on the model object so process_one can find it.
    head._calibration = ckpt.get("calibration")
    return head, hidden


def apply_calibration(probs: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Apply post-hoc Platt scaling if calibration is present in the checkpoint."""
    if not calibration:
        return probs
    if calibration.get("type") != "platt":
        return probs
    a = float(calibration["a"]); b = float(calibration["b"])
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def peak_events(probs: np.ndarray,
                min_prominence: float = 0.12,
                min_distance_sec: float = 1.0,
                rel_height: float = 0.6,
                smooth_ms: float = 80.0,
                hop_sec: float = HOP_SEC) -> list[dict]:
    """Peak-detection event extraction.

    This is the right algorithm when the model is under-confident (e.g. max
    breath probability ~0.35 on a clip that clearly has breaths). Threshold
    sweeping gives long blurry events because the prob curve sits near the
    threshold for most of the clip; peaks pull out the *locations* the model
    is actually firing at.

    Algorithm:
      1. Smooth the probability curve (~80 ms moving avg) to remove jitter.
      2. Find local maxima with at least `min_prominence` height-above-baseline
         and at least `min_distance_sec` apart.
      3. For each peak, walk outward to where prob drops below `rel_height` of
         the peak's height (above the curve's median). That's the event width.

    Returns events sorted by start time.
    """
    if len(probs) == 0:
        return []

    # Smooth
    w = max(1, int(round(smooth_ms / 1000.0 / hop_sec)))
    if w > 1:
        kernel = np.ones(w, dtype=np.float32) / w
        smooth = np.convolve(probs, kernel, mode="same")
    else:
        smooth = probs.copy()

    # Find peaks
    try:
        from scipy.signal import find_peaks
    except ImportError:
        # Fallback: simple local-max scan (scipy is a hard dep so this rarely fires)
        peaks = [i for i in range(1, len(smooth) - 1)
                 if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
                 and smooth[i] > min_prominence]
        peaks = np.asarray(peaks, dtype=int)
    else:
        min_distance_frames = max(1, int(round(min_distance_sec / hop_sec)))
        peaks, _ = find_peaks(
            smooth,
            prominence=min_prominence,
            distance=min_distance_frames,
        )

    out = []
    baseline = float(np.median(smooth))
    for p in peaks:
        peak_val = float(smooth[p])
        cutoff = baseline + (peak_val - baseline) * rel_height

        # Walk left
        l = p
        while l > 0 and smooth[l - 1] >= cutoff:
            l -= 1
        # Walk right
        r = p
        while r < len(smooth) - 1 and smooth[r + 1] >= cutoff:
            r += 1

        out.append({
            "start_sec": round(l * hop_sec, 3),
            "end_sec":   round((r + 1) * hop_sec, 3),  # +1 for half-open interval
            "score":     round(peak_val, 3),
        })
    return out


def threshold_events(probs: np.ndarray, threshold: float,
                     min_dur_sec: float = 0.05,
                     merge_gap_sec: float = 0.20,
                     smooth_ms: float = 50.0,
                     hop_sec: float = HOP_SEC) -> list[dict]:
    """Threshold + smoothing + min-duration + merge-close-events post-processing.

    Two consecutive events whose gap (start2 - end1) is below merge_gap_sec are
    merged into one. This handles the common case where the breath probability
    dips briefly below threshold in the middle of a real breath, splitting it
    into two spurious events.

    `smooth_ms` applies a centered moving-average smoothing to the probability
    curve before thresholding. 50 ms is 5 frames at 10 ms hop — enough to bridge
    the small per-frame dips that fragment a single real breath, without
    blurring closely-spaced real breaths together.
    """
    # Smooth probabilities first (centered moving average)
    if smooth_ms > 0 and len(probs) > 0:
        w = max(1, int(round(smooth_ms / 1000.0 / hop_sec)))
        if w > 1:
            kernel = np.ones(w, dtype=np.float32) / w
            probs = np.convolve(probs, kernel, mode="same")

    binary = probs >= threshold
    raw = []
    in_event = False
    start = 0
    for i, b in enumerate(binary):
        if b and not in_event:
            in_event = True
            start = i
        elif not b and in_event:
            in_event = False
            raw.append((start, i))
    if in_event:
        raw.append((start, len(binary)))

    # Merge events with small gaps between them
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) * hop_sec < merge_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    out = []
    for s, e in merged:
        dur = (e - s) * hop_sec
        if dur < min_dur_sec:
            continue
        out.append({
            "start_sec": round(s * hop_sec, 3),
            "end_sec":   round(e * hop_sec, 3),
            "score":     round(float(probs[s:e].max()), 3),
        })
    return out


def derive_phrase_events(breath_events: list[dict], duration_sec: float,
                         min_phrase_sec: float = 0.5) -> list[dict]:
    """Phrases = gaps between breath events. Sub-min_phrase_sec phrases are
    filtered out (they're typically the artifact of a noisy detector splitting
    a single breath into two)."""
    if not breath_events:
        return [{"start_sec": 0.0, "end_sec": round(duration_sec, 3),
                 "duration_sec": round(duration_sec, 3)}]

    phrases = []
    prev_end = 0.0
    for ev in breath_events:
        if ev["start_sec"] > prev_end:
            dur = ev["start_sec"] - prev_end
            if dur >= min_phrase_sec:
                phrases.append({
                    "start_sec":   round(prev_end, 3),
                    "end_sec":     round(ev["start_sec"], 3),
                    "duration_sec": round(dur, 3),
                })
        prev_end = ev["end_sec"]
    if prev_end < duration_sec:
        dur = duration_sec - prev_end
        if dur >= min_phrase_sec:
            phrases.append({
                "start_sec":   round(prev_end, 3),
                "end_sec":     round(duration_sec, 3),
                "duration_sec": round(dur, 3),
            })
    return phrases


def render_spectrogram_png(mel: np.ndarray, output_path: Path,
                           width_px: int = 1600, height_px: int = 280):
    """Render log-mel spectrogram as a borderless PNG (viridis colormap)."""
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(mel.T, aspect="auto", origin="lower",
              cmap="magma", interpolation="nearest")
    fig.savefig(output_path, dpi=100, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def process_one(audio_path: Path, output_dir: Path,
                nanopitch, head: BreathHead, hidden: int,
                detector: RuinskiyDetector,
                threshold: float, phrases_from: str,
                method: str = "peak",
                peak_prominence: float = 0.05,
                peak_min_distance_sec: float = 0.30,
                verbose: bool = True) -> Path:
    """Process one WAV — writes JSON + PNG + symlinked WAV into output_dir.

    Returns the JSON path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    json_path = output_dir / f"{stem}.json"
    png_path = output_dir / f"{stem}.png"
    wav_link = output_dir / f"{stem}.wav"

    if not wav_link.exists():
        wav_link.symlink_to(audio_path.resolve())

    if verbose:
        print(f"[{stem}] loading audio...")
    clip = load_labeled_clip(audio_path)
    waveform = clip.waveform
    duration = len(waveform) / SAMPLE_RATE

    if verbose:
        print(f"[{stem}] {duration:.2f}s @ {SAMPLE_RATE} Hz")

    mel = compute_log_mel(waveform)
    mel_t = torch.from_numpy(mel).unsqueeze(0)

    joint = JointModel(nanopitch, head)
    joint.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        vad, pitch, breath = joint(mel_t)
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0
    per_frame_ms = inference_ms / max(1, mel.shape[0])

    vad = vad.squeeze().numpy()
    pitch_argmax = pitch.squeeze().numpy().argmax(axis=-1).astype(np.float32)
    pitch_norm = (pitch_argmax / 360.0).tolist()
    breath_prob = breath.squeeze().numpy()
    # Apply Platt calibration if it's stored in the checkpoint
    breath_prob = apply_calibration(breath_prob, getattr(head, "_calibration", None))

    ruinskiy_events = [
        {"start_sec": round(e.start_sec, 3), "end_sec": round(e.end_sec, 3),
         "score": round(float(e.score), 3)}
        for e in detector.detect_array(waveform.astype(np.float32), SAMPLE_RATE)
    ]

    if method == "peak":
        predicted_events = peak_events(
            breath_prob,
            min_prominence=peak_prominence,
            min_distance_sec=peak_min_distance_sec,
        )
    else:
        predicted_events = threshold_events(breath_prob, threshold)
    phrase_source = predicted_events if phrases_from == "breath_head" else ruinskiy_events
    phrase_events = derive_phrase_events(phrase_source, duration)

    render_spectrogram_png(mel, png_path)

    out = {
        "audio_file": wav_link.name,
        "spectrogram_file": png_path.name,
        "duration_sec": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "hop_sec": HOP_SEC,
        "n_frames": int(len(breath_prob)),
        "breath_prob": [round(float(x), 4) for x in breath_prob],
        "voiced_prob": [round(float(x), 4) for x in vad],
        "pitch_norm":  pitch_norm,
        "predicted_events": predicted_events,
        "ruinskiy_events":  ruinskiy_events,
        "phrase_events":    phrase_events,
        "threshold": threshold,
        "phrases_from": phrases_from,
        "model_meta": {
            "hidden": hidden,
            "params": int(sum(p.numel() for p in head.parameters())),
            "inference_ms": round(inference_ms, 2),
            "per_frame_ms": round(per_frame_ms, 3),
        },
    }
    json_path.write_text(json.dumps(out))

    if verbose:
        kb = json_path.stat().st_size / 1024
        print(f"[{stem}] → {json_path.name} ({kb:.1f} KB) + spectrogram "
              f"(inference {inference_ms:.1f} ms total, {per_frame_ms:.3f} ms/frame)")
        print(f"[{stem}] predicted={len(predicted_events)}  "
              f"ruinskiy={len(ruinskiy_events)}  phrases={len(phrase_events)}")
    return json_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nanopitch", type=Path, required=True)
    p.add_argument("--breath-head", type=Path, required=True)
    p.add_argument("--audio", type=Path, help="single WAV (use --audio-dir for batch)")
    p.add_argument("--audio-dir", type=Path, help="directory of WAVs to batch-process")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--method", choices=["peak", "threshold"], default="peak",
                   help="Event extraction: 'peak' (find local maxima — robust to "
                        "under-confident models) or 'threshold' (cross threshold "
                        "and merge gaps — works only when model outputs sharp peaks).")
    p.add_argument("--peak-prominence", type=float, default=0.12,
                   help="Min prominence for peak detection (relative to local baseline).")
    p.add_argument("--peak-min-distance-sec", type=float, default=1.0,
                   help="Min seconds between two peaks (real singing breaths are ~1+s apart).")
    p.add_argument("--phrases-from", choices=["breath_head", "ruinskiy"],
                   default="ruinskiy")
    args = p.parse_args()

    if not args.audio and not args.audio_dir:
        p.error("Provide either --audio or --audio-dir")

    print("Loading models...")
    nanopitch = load_nanopitch(args.nanopitch)
    head, hidden = load_breath_head(args.breath_head)
    detector = RuinskiyDetector()
    print(f"BreathHead: {hidden} hidden, "
          f"{sum(p.numel() for p in head.parameters()):,} params")

    kw = dict(method=args.method,
              peak_prominence=args.peak_prominence,
              peak_min_distance_sec=args.peak_min_distance_sec)
    if args.audio:
        process_one(args.audio, args.output_dir, nanopitch, head, hidden,
                    detector, args.threshold, args.phrases_from, **kw)
    else:
        wavs = sorted(args.audio_dir.glob("*.wav"))
        print(f"Batch: {len(wavs)} WAVs")
        for w in wavs:
            try:
                process_one(w, args.output_dir, nanopitch, head, hidden,
                            detector, args.threshold, args.phrases_from, **kw)
            except Exception as exc:
                print(f"[{w.stem}] ERROR: {exc}")

        # Write a manifest the browser can read to list available clips
        manifest = []
        for jf in sorted(args.output_dir.glob("*.json")):
            if jf.stem == "manifest":
                continue
            d = json.loads(jf.read_text())
            manifest.append({
                "id": jf.stem,
                "name": jf.stem.replace("_", " "),
                "duration_sec": d["duration_sec"],
                "json": jf.name,
                "audio": d["audio_file"],
                "spectrogram": d["spectrogram_file"],
                "n_predicted": len(d["predicted_events"]),
                "n_ruinskiy":  len(d["ruinskiy_events"]),
                "n_phrases":   len(d["phrase_events"]),
            })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest))
        print(f"\nWrote manifest with {len(manifest)} entries.")


if __name__ == "__main__":
    main()
