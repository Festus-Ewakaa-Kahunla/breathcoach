#!/usr/bin/env python3
"""
VocalSet-Breath in-browser labeler — local HTTP server.

Endpoints
  GET  /                       index.html (the labeler UI)
  GET  /api/clips              list all clips with status (labeled / draft / untouched)
  GET  /api/clips/<id>.wav     audio (16 kHz mono, served with Range support)
  GET  /api/clips/<id>.png     pre-rendered log-mel spectrogram
  GET  /api/labels/<id>        existing .breath.json OR fresh model pre-fill
  POST /api/labels/<id>        save .breath.json

Run:
  cd dataset/tools/labeler
  python server.py --audio-dir ../../../data/_sung_excerpts \
                   --labels-dir ../../labels/vocalset \
                   --breath-head ../../../runs/v11-finetune-gtsinger/best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pathlib as _pathlib
torch.serialization.add_safe_globals([_pathlib.PosixPath, _pathlib.PurePosixPath])

from nanobreath.deployment.precompute_predictions import (  # noqa: E402
    load_nanopitch, load_breath_head, peak_events, apply_calibration,
    render_spectrogram_png,
)
from nanobreath.model.joint import JointModel                              # noqa: E402
from nanobreath.data.dataset import (                                       # noqa: E402
    load_labeled_clip, compute_log_mel, compute_breath_features,
)


# ─── Globals (loaded once at startup) ───────────────────────────────────────
_nano = None
_head = None
_audio_dir: Path | None = None
_labels_dir: Path | None = None
_spec_cache: Path | None = None
_prefill_prominence: float = 0.22  # tuned for VocalSet on v11


# ─── Model + label helpers ──────────────────────────────────────────────────
def load_models(nano_path: Path, head_path: Path) -> None:
    global _nano, _head
    print(f"  loading NanoPitch from {nano_path.name}...")
    _nano = load_nanopitch(nano_path)
    print(f"  loading BreathHead from {head_path.parent.name}/{head_path.name}...")
    _head, _ = load_breath_head(head_path)
    if getattr(_head, "_backbone_state", None) is not None:
        _nano.load_state_dict(_head._backbone_state)
        print(f"  swapped in finetuned backbone shipped with {head_path.parent.name}")


def list_clips() -> list[dict]:
    """All wavs in audio_dir, annotated with label status + counts."""
    out = []
    for wav in sorted(_audio_dir.glob("*.wav")):
        stem = wav.stem
        label_path = _labels_dir / f"{stem}.breath.json"
        try:
            with wave.open(str(wav), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
        except Exception:
            duration = 0.0
        status, n_breath, n_hard, n_total, n_candidates = "untouched", 0, 0, 0, 0
        if label_path.exists():
            d = json.loads(label_path.read_text())
            n_breath = len(d.get("breath_events", []))
            n_hard = len(d.get("hard_negatives", []))
            n_total = sum(len(d.get(k, [])) for k in
                          ("breath_events", "silent_breaths", "uncertain",
                           "hard_negatives", "exhales"))
            # Unreviewed seeded hard-negatives still awaiting an A/B/U/D decision.
            n_candidates = sum(1 for ev in d.get("hard_negatives", [])
                               if ev.get("source") == "v13_low_thresh_candidate")
            # File exists at all = user explicitly saved = labeled. Per-event
            # provenance (v11_prefill vs human_*) is tracked in the `source`
            # field of each event and reported in the paper; it doesn't change
            # the clip-level status.
            status = "labeled"
        out.append({
            "id": stem,
            "filename": wav.name,
            "duration_sec": round(duration, 2),
            "status": status,
            "n_breath": n_breath,
            "n_hard_negative": n_hard,
            "n_total": n_total,
            "n_candidates_remaining": n_candidates,
        })
    return out


def _is_all_draft(label: dict) -> bool:
    for cat in ("breath_events", "silent_breaths", "uncertain", "hard_negatives", "exhales"):
        for ev in label.get(cat, []):
            if not ev.get("source", "").startswith("v11"):
                return False
    return True


def generate_prefill(stem: str) -> dict:
    """Run v11 on a clip and wrap the events in our schema, marked as draft."""
    wav_path = _audio_dir / f"{stem}.wav"
    clip = load_labeled_clip(wav_path)
    mel = compute_log_mel(clip.waveform)
    extra_t = None
    if getattr(_head, "_use_extra", False):
        feats = compute_breath_features(clip.waveform, mel)
        extra_t = torch.from_numpy(feats).unsqueeze(0)
    joint = JointModel(_nano, _head)
    joint.eval()
    with torch.no_grad():
        _vad, _pitch, breath = joint(torch.from_numpy(mel).unsqueeze(0),
                                     extra_features=extra_t)
    bp = apply_calibration(breath.squeeze().numpy(),
                           getattr(_head, "_calibration", None))
    events = peak_events(bp, min_prominence=_prefill_prominence, min_distance_sec=1.0)
    duration = len(clip.waveform) / 16000
    return {
        "audio_file": f"{stem}.wav",
        "sample_rate": 16000,
        "duration_sec": round(duration, 4),
        "labeler": "",
        "label_date": "",
        "tool_version": "breathcoach-labeler 0.1",
        "review_time_sec": 0,
        "pre_fill_source": "v11-finetune-gtsinger",
        "pre_fill_prominence": _prefill_prominence,
        "breath_events": [
            {"start_sec": float(e["start_sec"]),
             "end_sec":   float(e["end_sec"]),
             "score":     float(e.get("score", 0)),
             "confidence": "medium",
             "source": "v11_prefill",
             "notes": ""}
            for e in events
        ],
        "silent_breaths": [],
        "uncertain":      [],
        "hard_negatives": [],
        "exhales":        [],
        "notes": "",
    }


def get_or_create_label(stem: str) -> dict:
    label_path = _labels_dir / f"{stem}.breath.json"
    if label_path.exists():
        return json.loads(label_path.read_text())
    return generate_prefill(stem)


def save_label(stem: str, data: dict) -> None:
    _labels_dir.mkdir(parents=True, exist_ok=True)
    (_labels_dir / f"{stem}.breath.json").write_text(json.dumps(data, indent=2))


def get_spectrogram_bytes(stem: str) -> bytes:
    png_path = _spec_cache / f"{stem}.png"
    if not png_path.exists():
        wav_path = _audio_dir / f"{stem}.wav"
        clip = load_labeled_clip(wav_path)
        mel = compute_log_mel(clip.waveform)
        render_spectrogram_png(mel, png_path)
    return png_path.read_bytes()


# ─── HTTP handler ───────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send_bytes(json.dumps(obj).encode("utf-8"), "application/json", code)

    def _send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        rng = self.headers.get("Range")
        if rng:
            try:
                spec = rng.split("=", 1)[1]
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                if start < 0 or start > end:
                    raise ValueError
            except (IndexError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers(); return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start); remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk: break
                    try: self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError): return
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    try: self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError): return

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send_file(Path(__file__).parent / "index.html", "text/html")
            elif self.path == "/api/clips":
                self._send_json(list_clips())
            elif self.path.startswith("/api/clips/") and self.path.endswith(".wav"):
                stem = self.path.split("/")[-1][:-4]
                wav = _audio_dir / f"{stem}.wav"
                if not wav.exists(): self.send_error(404); return
                self._send_file(wav, "audio/wav")
            elif self.path.startswith("/api/clips/") and self.path.endswith(".png"):
                stem = self.path.split("/")[-1][:-4]
                self._send_bytes(get_spectrogram_bytes(stem), "image/png")
            elif self.path.startswith("/api/labels/"):
                stem = self.path.split("/")[-1]
                self._send_json(get_or_create_label(stem))
            else:
                self.send_error(404)
        except Exception as exc:
            import traceback; traceback.print_exc()
            try: self._send_json({"error": str(exc)}, 500)
            except Exception: pass

    def do_POST(self):
        if self.path.startswith("/api/labels/"):
            stem = self.path.split("/")[-1]
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            try:
                save_label(stem, json.loads(body))
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        msg = format % args
        if "POST" in msg or any(c in msg for c in (" 5", " 4")):
            sys.stderr.write(f"{self.address_string()} {msg}\n")


# ─── Entrypoint ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8431)
    ap.add_argument("--audio-dir",  type=Path,
                    default=REPO_ROOT / "data" / "_sung_excerpts")
    ap.add_argument("--labels-dir", type=Path,
                    default=REPO_ROOT / "dataset" / "labels" / "vocalset")
    ap.add_argument("--spec-cache", type=Path,
                    default=REPO_ROOT / "dataset" / "tools" / "labeler" / ".spec_cache")
    ap.add_argument("--nanopitch",  type=Path,
                    default=REPO_ROOT / "models" / "nanopitch" / "best.pth")
    ap.add_argument("--breath-head", type=Path,
                    default=REPO_ROOT / "runs" / "v11-finetune-gtsinger" / "best.pth")
    ap.add_argument("--prefill-prominence", type=float, default=0.22,
                    help="peak_events min_prominence for v11 draft markers")
    args = ap.parse_args()

    global _audio_dir, _labels_dir, _spec_cache, _prefill_prominence
    _audio_dir = args.audio_dir.resolve()
    _labels_dir = args.labels_dir.resolve()
    _spec_cache = args.spec_cache.resolve()
    _prefill_prominence = float(args.prefill_prominence)
    _labels_dir.mkdir(parents=True, exist_ok=True)
    _spec_cache.mkdir(parents=True, exist_ok=True)

    if not _audio_dir.exists():
        sys.exit(f"audio dir not found: {_audio_dir}")
    if not args.nanopitch.exists():
        sys.exit(f"nanopitch checkpoint not found: {args.nanopitch}")
    if not args.breath_head.exists():
        sys.exit(f"breath-head checkpoint not found: {args.breath_head}")

    print("Loading models...")
    load_models(args.nanopitch, args.breath_head)

    n = len(list(_audio_dir.glob("*.wav")))
    print(f"\nAudio dir:  {_audio_dir} ({n} clips)")
    print(f"Labels dir: {_labels_dir}")
    print(f"Cache dir:  {_spec_cache}")
    print(f"Pre-fill prominence: {_prefill_prominence}")
    print(f"\nLabeler running at http://localhost:{args.port}/")
    print("Ctrl+C to stop\n")

    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\nStopped.")
