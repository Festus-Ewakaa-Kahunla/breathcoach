#!/usr/bin/env python3
"""
Ruinskiy & Lavner 2007 — MFCC + Template-Matching Breath Detector.

Reference:
    Ruinskiy, D., & Lavner, Y. (2007). An effective algorithm for automatic
    detection and exact demarcation of breath sounds in speech and song signals.
    IEEE Transactions on Audio, Speech, and Language Processing, 15(3), 838-850.
    https://ieeexplore.ieee.org/document/4100696

What this module does:
- Implements the canonical breath-detection algorithm: extract MFCCs, build a
  reference template from labeled breath examples (or use a generic one), and
  detect breaths by sliding-window correlation against the template.
- Frames detected breath segments by post-processing the correlation score
  with thresholding + minimum-duration constraints.
- Outputs a list of (start_sec, end_sec) breath events per audio file.

Why we have it:
1. Published baseline for the paper. We compare our neural head against it.
2. Pseudo-label generator: run on unlabeled VocalSet to produce weak labels
   for training augmentation.
3. Plan B fallback: if our neural head fails to deploy in WASM, this DSP
   algorithm runs in pure JS for a degraded but functional demo.

Note: this is a faithful re-implementation of the algorithm as described in the
paper, NOT a port of any author code (the original is in MATLAB, not public).
We deviate only where the paper is ambiguous; deviations are flagged with
PAPER-NOTE comments.

Usage:
    from ruinskiy_lavner import RuinskiyDetector
    det = RuinskiyDetector()  # uses default generic template
    events = det.detect(wav_path)
    # events: list of (start_sec, end_sec) tuples
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Default detector parameters from the paper (Section IV / Table I).
DEFAULTS = dict(
    sample_rate=16000,
    frame_length=0.025,        # 25 ms analysis window
    hop_length=0.010,          # 10 ms hop
    n_mfcc=13,                 # MFCC coefficients incl. C0
    n_mels=40,                 # mel filter bank
    fft_size=512,              # at 16 kHz, 32 ms — covers our 25 ms window
    correlation_threshold=0.5, # threshold on normalized cross-correlation
    min_breath_duration=0.05,  # 50 ms min breath length
    max_breath_duration=1.0,   # 1.0 s max breath length (safety)
    template_duration=0.30,    # 300 ms generic template
)


@dataclass
class BreathEvent:
    start_sec: float
    end_sec: float
    score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _frame_to_sec(frame_idx: int, hop_sec: float) -> float:
    return frame_idx * hop_sec


def _stft_magnitude(signal: np.ndarray, fft_size: int, hop_samples: int,
                    frame_samples: int) -> np.ndarray:
    """Compute magnitude STFT using a Hann window. Output shape (freq, time)."""
    window = np.hanning(frame_samples)
    n_frames = max(0, (len(signal) - frame_samples) // hop_samples + 1)
    if n_frames <= 0:
        return np.zeros((fft_size // 2 + 1, 0), dtype=np.float32)
    out = np.empty((fft_size // 2 + 1, n_frames), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_samples
        frame = signal[start:start + frame_samples] * window
        spec = np.fft.rfft(frame, n=fft_size)
        out[:, t] = np.abs(spec).astype(np.float32)
    return out


def _mel_filterbank(n_mels: int, fft_size: int, sample_rate: int,
                    f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    """Standard HTK-style triangular mel filterbank. Shape (n_mels, fft_size//2+1)."""
    if f_max is None:
        f_max = sample_rate / 2

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = fft_size // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(max(0, left), min(n_bins, center)):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(max(0, center), min(n_bins, right)):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def compute_mfcc(signal: np.ndarray, sample_rate: int = 16000,
                 n_mfcc: int = 13, n_mels: int = 40, fft_size: int = 512,
                 frame_length: float = 0.025, hop_length: float = 0.010,
                 skip_c0: bool = True) -> np.ndarray:
    """Compute MFCCs from a mono float32 signal. Output shape (n_mfcc, n_frames).

    PAPER-NOTE: We skip the C0 (DC/energy) coefficient by default. C0 captures
    overall log-energy, which makes singing-vs-breath strongly anti-correlated
    on this axis (singing is loud, breath is quiet). Including C0 dominates the
    correlation and washes out the spectral-shape match we actually want.
    Setting skip_c0=False reverts to "raw" Ruinskiy-style MFCCs.
    """
    frame_samples = int(round(frame_length * sample_rate))
    hop_samples = int(round(hop_length * sample_rate))
    mag = _stft_magnitude(signal, fft_size, hop_samples, frame_samples)  # (F, T)
    if mag.shape[1] == 0:
        return np.zeros((n_mfcc, 0), dtype=np.float32)
    fb = _mel_filterbank(n_mels, fft_size, sample_rate)                  # (M, F)
    mel_energy = fb @ (mag ** 2)                                          # (M, T)
    log_mel = np.log(mel_energy + 1e-10)                                  # (M, T)
    # DCT-II to get MFCCs. We compute it manually so we don't depend on scipy.
    M = log_mel.shape[0]
    # If skip_c0, we ask for n_mfcc+1 coefficients then drop the first.
    n_coef = n_mfcc + (1 if skip_c0 else 0)
    n = np.arange(M).reshape(-1, 1)
    k = np.arange(n_coef).reshape(1, -1)
    dct_basis = np.cos(np.pi / M * (n + 0.5) * k)  # (M, n_coef)
    mfcc = dct_basis.T @ log_mel                    # (n_coef, T)
    if skip_c0:
        mfcc = mfcc[1:]  # drop C0
    return mfcc.astype(np.float32)


def normalized_cross_correlation(template: np.ndarray, signal_mfcc: np.ndarray) -> np.ndarray:
    """Slide template over signal_mfcc and return per-frame correlation.

    template:    (n_mfcc, t_template)
    signal_mfcc: (n_mfcc, t_signal)

    Returns: (t_signal,) — correlation centered at each output frame. Frames where
    the template overflows the signal boundary get score 0.
    """
    n_mfcc, t_t = template.shape
    _, t_s = signal_mfcc.shape
    if t_t == 0 or t_s == 0:
        return np.zeros(t_s, dtype=np.float32)

    # Flatten template to a vector for correlation.
    tmpl_flat = template.reshape(-1)
    tmpl_norm = np.linalg.norm(tmpl_flat) + 1e-10

    out = np.zeros(t_s, dtype=np.float32)
    half = t_t // 2
    for t in range(t_s):
        start = t - half
        end = start + t_t
        if start < 0 or end > t_s:
            continue
        win = signal_mfcc[:, start:end].reshape(-1)
        win_norm = np.linalg.norm(win) + 1e-10
        out[t] = float(np.dot(tmpl_flat, win) / (tmpl_norm * win_norm))
    return out


def detect_events_from_score(score: np.ndarray, hop_sec: float,
                             threshold: float, min_dur_sec: float,
                             max_dur_sec: float) -> List[BreathEvent]:
    """Threshold the per-frame correlation and post-process into events.

    Algorithm:
      1. Threshold to get binary mask.
      2. Find contiguous true-runs.
      3. Drop runs shorter than min_dur or longer than max_dur (latter as safety).
      4. Each surviving run becomes one BreathEvent with mean score.
    """
    if len(score) == 0:
        return []
    mask = score >= threshold
    events: List[BreathEvent] = []
    in_run = False
    run_start = 0
    for t in range(len(mask)):
        if mask[t] and not in_run:
            in_run = True
            run_start = t
        elif not mask[t] and in_run:
            in_run = False
            _maybe_emit(events, score, run_start, t, hop_sec, min_dur_sec, max_dur_sec)
    if in_run:
        _maybe_emit(events, score, run_start, len(mask), hop_sec, min_dur_sec, max_dur_sec)
    return events


def _maybe_emit(events: List[BreathEvent], score: np.ndarray,
                start: int, end: int, hop_sec: float,
                min_dur_sec: float, max_dur_sec: float) -> None:
    dur = (end - start) * hop_sec
    if dur < min_dur_sec or dur > max_dur_sec:
        return
    mean_score = float(np.mean(score[start:end]))
    events.append(BreathEvent(
        start_sec=_frame_to_sec(start, hop_sec),
        end_sec=_frame_to_sec(end, hop_sec),
        score=mean_score,
    ))


class RuinskiyDetector:
    """Singing-breath detector following Ruinskiy & Lavner 2007.

    Usage:
        det = RuinskiyDetector()
        events = det.detect_array(audio_signal, sample_rate=16000)
        # or
        events = det.detect_file('path/to/audio.wav')
    """

    def __init__(self, **kwargs):
        # Merge user kwargs over DEFAULTS
        cfg = {**DEFAULTS, **kwargs}
        self.sample_rate = cfg["sample_rate"]
        self.frame_length = cfg["frame_length"]
        self.hop_length = cfg["hop_length"]
        self.n_mfcc = cfg["n_mfcc"]
        self.n_mels = cfg["n_mels"]
        self.fft_size = cfg["fft_size"]
        self.threshold = cfg["correlation_threshold"]
        self.min_dur = cfg["min_breath_duration"]
        self.max_dur = cfg["max_breath_duration"]
        self.template_duration = cfg["template_duration"]
        self.template: np.ndarray | None = None  # set via fit() or use generic

    def fit_template(self, breath_clips: List[np.ndarray]) -> None:
        """Build a template by averaging MFCCs of provided breath examples.

        Each clip in breath_clips should be a 1D float32 array containing a
        single isolated breath. The clip length should be roughly
        self.template_duration seconds.
        """
        if len(breath_clips) == 0:
            raise ValueError("Need at least one breath clip to fit a template.")
        target_frames = int(round(self.template_duration / self.hop_length))
        accumulated = []
        for clip in breath_clips:
            mfcc = compute_mfcc(
                clip, self.sample_rate, self.n_mfcc, self.n_mels,
                self.fft_size, self.frame_length, self.hop_length,
            )
            if mfcc.shape[1] == 0:
                continue
            # PAPER-NOTE: paper says "average MFCCs over fixed-length windows".
            # We resample each clip's MFCC time axis to target_frames via simple
            # linear interpolation, which is equivalent and language-agnostic.
            mfcc = _resample_time(mfcc, target_frames)
            accumulated.append(mfcc)
        if not accumulated:
            raise ValueError("All clips were too short to extract MFCC frames.")
        self.template = np.mean(np.stack(accumulated, axis=0), axis=0).astype(np.float32)

    def use_generic_template(self) -> None:
        """Synthesize a generic breath template from white noise + spectral shaping.

        PAPER-NOTE: we use this when no labeled examples are yet available.
        Realistic enough to bootstrap: real breaths peak at 1.5-2.5 kHz, mostly
        broadband mid-frequency noise.
        """
        rng = np.random.default_rng(seed=42)
        clip_len_sec = self.template_duration
        n_samples = int(round(clip_len_sec * self.sample_rate))
        # Pink-ish noise: white * 1/sqrt(f) shaping in spectral domain
        white = rng.standard_normal(n_samples)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sample_rate)
        # Bandpass-emphasis 800 Hz - 4 kHz where breath energy concentrates
        emphasis = np.exp(-((np.log(np.maximum(freqs, 1)) - np.log(2000)) ** 2) / 2.0)
        spec *= emphasis
        synthetic = np.fft.irfft(spec, n=n_samples).astype(np.float32)
        synthetic /= (np.max(np.abs(synthetic)) + 1e-10)
        self.fit_template([synthetic])

    def detect_array(self, signal: np.ndarray, sample_rate: int) -> List[BreathEvent]:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Signal at {sample_rate} Hz; detector configured for {self.sample_rate}. "
                f"Resample before calling."
            )
        if self.template is None:
            self.use_generic_template()
        mfcc = compute_mfcc(
            signal, self.sample_rate, self.n_mfcc, self.n_mels,
            self.fft_size, self.frame_length, self.hop_length,
        )
        score = normalized_cross_correlation(self.template, mfcc)
        events = detect_events_from_score(
            score, hop_sec=self.hop_length,
            threshold=self.threshold,
            min_dur_sec=self.min_dur,
            max_dur_sec=self.max_dur,
        )
        return events

    def detect_file(self, wav_path: str | Path) -> List[BreathEvent]:
        signal, sr = _read_wav_mono(wav_path)
        if sr != self.sample_rate:
            signal = _resample_audio(signal, sr, self.sample_rate)
        return self.detect_array(signal.astype(np.float32), self.sample_rate)


def _resample_time(arr: np.ndarray, target_t: int) -> np.ndarray:
    """Linear-interpolate array along its second (time) axis to target_t frames."""
    n_feat, t = arr.shape
    if t == target_t:
        return arr
    if t == 0 or target_t == 0:
        return np.zeros((n_feat, target_t), dtype=arr.dtype)
    src_idx = np.linspace(0, t - 1, num=target_t)
    out = np.empty((n_feat, target_t), dtype=arr.dtype)
    for i in range(n_feat):
        out[i, :] = np.interp(src_idx, np.arange(t), arr[i, :])
    return out


def _read_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in [-1, 1]. Uses scipy if available, else wave."""
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)
        return data, int(sr)
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n_frames = w.getnframes()
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            raw = w.readframes(n_frames)
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        return data, sr


def _resample_audio(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lazy linear resampling. Good enough for MFCC-based features."""
    if src_sr == dst_sr:
        return signal
    ratio = dst_sr / src_sr
    n_dst = int(round(len(signal) * ratio))
    src_idx = np.linspace(0, len(signal) - 1, num=n_dst)
    return np.interp(src_idx, np.arange(len(signal)), signal).astype(np.float32)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ruinskiy & Lavner breath detector")
    parser.add_argument("wav", type=Path, help="WAV file to analyze")
    parser.add_argument("--threshold", type=float, default=DEFAULTS["correlation_threshold"])
    parser.add_argument("--min-dur", type=float, default=DEFAULTS["min_breath_duration"])
    parser.add_argument("--out-json", type=Path, default=None,
                       help="optional path to write events as JSON")
    args = parser.parse_args()

    det = RuinskiyDetector(
        correlation_threshold=args.threshold,
        min_breath_duration=args.min_dur,
    )
    events = det.detect_file(args.wav)
    print(f"Detected {len(events)} breath events in {args.wav}")
    for ev in events:
        print(f"  {ev.start_sec:7.3f} - {ev.end_sec:7.3f} s  ({ev.duration_sec*1000:5.1f} ms, score={ev.score:.3f})")

    if args.out_json:
        out = {
            "audio_file": str(args.wav),
            "detector": "ruinskiy_lavner_2007",
            "threshold": args.threshold,
            "events": [
                {"start_sec": ev.start_sec, "end_sec": ev.end_sec, "score": ev.score}
                for ev in events
            ],
        }
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
