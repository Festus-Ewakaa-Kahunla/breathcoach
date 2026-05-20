"""
Dataset loader for breath-event detection.

Reads (audio.wav, audio.breath.json) pairs and produces frame-level breath labels
aligned with NanoPitch's mel spectrogram input.

Frame-rate convention (matches NanoPitch):
- Audio at 16 kHz
- Mel hop = 10 ms (160 samples per frame)
- Mel window = 25 ms (400 samples per frame)
- 40 mel bands

A breath event labeled (start_sec=1.83, end_sec=2.06) becomes:
- breath[183:206] = 1 in the frame-level binary label array (one frame per 10 ms)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


SAMPLE_RATE = 16000
HOP_SAMPLES = 160      # 10 ms hop
WINDOW_SAMPLES = 400   # 25 ms window
N_MELS = 40


@dataclass
class LabeledClip:
    """One clip + its breath labels, frame-aligned."""
    audio_path: Path
    sample_rate: int
    waveform: np.ndarray         # (n_samples,) float32 [-1, 1]
    breath_labels: np.ndarray    # (n_frames,) float32 in {0.0, 1.0}
    breath_events: list          # original list of {start_sec, end_sec, ...}
    confidence: np.ndarray       # (n_frames,) float32, 1.0 unless event marked low

    @property
    def n_frames(self) -> int:
        return len(self.breath_labels)

    @property
    def duration_sec(self) -> float:
        return self.n_frames * (HOP_SAMPLES / SAMPLE_RATE)


def load_labeled_clip(audio_path: Path, label_path: Optional[Path] = None) -> LabeledClip:
    """Load a WAV + its sibling .breath.json into a LabeledClip.

    If label_path is None, looks for `<audio>.breath.json` next to the WAV.
    If no label JSON exists, returns a clip with all-zero breath labels.
    """
    audio_path = Path(audio_path)
    if label_path is None:
        label_path = audio_path.with_suffix(".breath.json")
    label_path = Path(label_path)

    waveform, sr = _read_wav_mono(audio_path)
    if sr != SAMPLE_RATE:
        waveform = _resample_audio(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE

    n_frames = max(0, (len(waveform) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    breath_labels = np.zeros(n_frames, dtype=np.float32)
    confidence = np.ones(n_frames, dtype=np.float32)
    breath_events = []

    if label_path.exists():
        with open(label_path) as f:
            label_data = json.load(f)
        breath_events = label_data.get("breath_events", [])
        for ev in breath_events:
            start_frame = int(round(ev["start_sec"] * SAMPLE_RATE / HOP_SAMPLES))
            end_frame = int(round(ev["end_sec"] * SAMPLE_RATE / HOP_SAMPLES))
            start_frame = max(0, min(n_frames, start_frame))
            end_frame = max(0, min(n_frames, end_frame))
            breath_labels[start_frame:end_frame] = 1.0
            if ev.get("confidence", "high") == "low":
                confidence[start_frame:end_frame] = 0.5

    return LabeledClip(
        audio_path=audio_path,
        sample_rate=SAMPLE_RATE,
        waveform=waveform.astype(np.float32),
        breath_labels=breath_labels,
        breath_events=breath_events,
        confidence=confidence,
    )


def compute_log_mel(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    n_mels: int = N_MELS, hop_samples: int = HOP_SAMPLES,
                    window_samples: int = WINDOW_SAMPLES,
                    fft_size: int = 512) -> np.ndarray:
    """Log-mel spectrogram in (n_frames, n_mels) layout matching NanoPitch input."""
    n_frames = max(0, (len(waveform) - window_samples) // hop_samples + 1)
    if n_frames == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    # Hann window once
    win = np.hanning(window_samples).astype(np.float32)

    # Mel filterbank (HTK-style)
    fb = _mel_filterbank(n_mels, fft_size, sample_rate)

    out = np.empty((n_frames, n_mels), dtype=np.float32)
    for t in range(n_frames):
        start = t * hop_samples
        frame = waveform[start:start + window_samples] * win
        spec = np.fft.rfft(frame, n=fft_size)
        mag2 = np.abs(spec) ** 2  # power
        mel = fb @ mag2
        out[t] = np.log(mel + 1e-10)
    return out.astype(np.float32)


def collect_labeled_clips(label_dir: Path) -> List[LabeledClip]:
    """Find all *.breath.json files in label_dir and return their LabeledClips.

    Audio must be alongside the JSON (same filename, .wav extension).
    """
    label_dir = Path(label_dir)
    clips = []
    for label_path in sorted(label_dir.rglob("*.breath.json")):
        audio_path = label_path.with_name(label_path.name.replace(".breath.json", ".wav"))
        if not audio_path.exists():
            print(f"WARN: {label_path} has no matching {audio_path}")
            continue
        clips.append(load_labeled_clip(audio_path, label_path))
    return clips


# ─── Private helpers (audio I/O + mel filterbank) ────────────────────────────

def _read_wav_mono(path: Path):
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
    if src_sr == dst_sr:
        return signal
    ratio = dst_sr / src_sr
    n_dst = int(round(len(signal) * ratio))
    src_idx = np.linspace(0, len(signal) - 1, num=n_dst)
    return np.interp(src_idx, np.arange(len(signal)), signal).astype(np.float32)


def _mel_filterbank(n_mels: int, fft_size: int, sample_rate: int,
                    f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
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


if __name__ == "__main__":
    # Quick smoke test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <audio.wav> [<labels.breath.json>]")
        sys.exit(0)
    audio = Path(sys.argv[1])
    labels = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    clip = load_labeled_clip(audio, labels)
    print(f"Audio: {clip.audio_path.name}")
    print(f"Duration: {clip.duration_sec:.2f}s ({clip.n_frames} frames)")
    print(f"Breath events labeled: {len(clip.breath_events)}")
    print(f"Frames marked breath: {int(clip.breath_labels.sum())} ({clip.breath_labels.mean()*100:.1f}%)")
    mel = compute_log_mel(clip.waveform)
    print(f"Mel shape: {mel.shape}")
