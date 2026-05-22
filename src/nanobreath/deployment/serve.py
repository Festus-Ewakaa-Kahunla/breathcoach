#!/usr/bin/env python3
"""
HTTP server with a /process endpoint that runs precompute_predictions inline.

Two routes:
  GET  /*               serve static files from the `web/` directory
  POST /process         accept an audio file upload, run NanoPitch+BreathHead+Ruinskiy,
                        return predictions JSON (same schema as the precomputed clips)

The browser uses /process so the user can record their own singing via the
MediaRecorder API and see live predictions — without needing a WASM build.

Usage:
    cd Project 2/prototypes/deployment
    python3 serve.py --port 8421 \
        --nanopitch /path/to/exp12-mixed-aug/checkpoints/best.pth \
        --breath-head /tmp/run_v2/best.pth

Notes:
- This is a single-threaded dev server; one request at a time. Fine for local demo.
- Model is loaded once at startup and held in memory; no per-request reload cost.
- Audio is decoded via soundfile if available, else falls back to scipy/wave.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import tempfile
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel
from nanobreath.data.dataset import compute_log_mel, SAMPLE_RATE, HOP_SAMPLES
from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector
from nanobreath.deployment.precompute_predictions import (
    threshold_events, peak_events, derive_phrase_events, render_spectrogram_png,
    load_nanopitch, load_breath_head, HOP_SEC,
)


# Loaded once at startup
_models = {"nanopitch": None, "head": None, "hidden": 8, "detector": None,
           "threshold": 0.25, "phrases_from": "breath_head"}


def decode_audio(blob: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV/WebM/OGG audio blob → (mono float32 samples in [-1,1], sample_rate)."""
    # Try wave first (assumes WAV/PCM)
    try:
        with wave.open(io.BytesIO(blob), "rb") as wf:
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        return data, sr
    except (wave.Error, EOFError):
        pass

    # Fallback: write to temp file and use soundfile (which uses libsndfile internally,
    # supports OGG/FLAC/WAV but NOT WebM/Opus). For WebM/Opus we'd need ffmpeg.
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(blob)
            tmp = f.name
        data, sr = sf.read(tmp)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode audio. Browser must record as WAV. "
            f"Error: {exc}"
        )


def resample_linear(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return signal
    ratio = dst_sr / src_sr
    n_dst = int(round(len(signal) * ratio))
    src_idx = np.linspace(0, len(signal) - 1, num=n_dst)
    return np.interp(src_idx, np.arange(len(signal)), signal).astype(np.float32)


def process_audio(waveform: np.ndarray, sample_rate: int,
                  spec_png_path: Path | None = None) -> dict:
    """Run the full inference pipeline on a waveform. Returns the same JSON schema
    as the precomputed clips so the browser can render it identically."""
    if sample_rate != SAMPLE_RATE:
        waveform = resample_linear(waveform, sample_rate, SAMPLE_RATE)
        sample_rate = SAMPLE_RATE

    duration = len(waveform) / sample_rate
    mel = compute_log_mel(waveform)
    mel_t = torch.from_numpy(mel).unsqueeze(0)

    joint = JointModel(_models["nanopitch"], _models["head"])
    joint.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        vad, pitch, breath = joint(mel_t)
    t1 = time.perf_counter()
    inference_ms = (t1 - t0) * 1000.0

    vad = vad.squeeze().numpy()
    pitch_argmax = pitch.squeeze().numpy().argmax(axis=-1).astype(np.float32)
    pitch_norm = (pitch_argmax / 360.0).tolist()
    breath_prob = breath.squeeze().numpy()

    ruinskiy_events = [
        {"start_sec": round(e.start_sec, 3), "end_sec": round(e.end_sec, 3),
         "score": round(float(e.score), 3)}
        for e in _models["detector"].detect_array(waveform.astype(np.float32), sample_rate)
    ]
    # Use peak detection by default — robust to under-confident model probs
    if _models.get("method", "peak") == "peak":
        predicted_events = peak_events(breath_prob)
    else:
        predicted_events = threshold_events(breath_prob, _models["threshold"])
    phrase_src = predicted_events if _models["phrases_from"] == "breath_head" else ruinskiy_events
    phrase_events = derive_phrase_events(phrase_src, duration)

    spec_file = None
    if spec_png_path is not None:
        render_spectrogram_png(mel, spec_png_path)
        spec_file = spec_png_path.name

    return {
        "audio_file": None,  # caller may overwrite
        "spectrogram_file": spec_file,
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
        "threshold": _models["threshold"],
        "phrases_from": _models["phrases_from"],
        "model_meta": {
            "hidden": _models["hidden"],
            "params": int(sum(p.numel() for p in _models["head"].parameters())),
            "inference_ms": round(inference_ms, 2),
            "per_frame_ms": round(inference_ms / max(1, mel.shape[0]), 3),
        },
    }


class Handler(SimpleHTTPRequestHandler):
    # HTTP/1.1 enables keep-alive. We also implement byte-range (206) responses
    # below, because browser <audio> elements require them to load a media file.
    # Python's stdlib SimpleHTTPRequestHandler ignores Range and replies 200 with
    # the whole file, which leaves the audio element stuck in readyState 0.
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.headers.get("Range"):
            self._serve_range()
        else:
            super().do_GET()

    def _serve_range(self):
        """Serve a 206 Partial Content response for a Range request."""
        import os as _os
        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found"); return
        try:
            fs = _os.fstat(f.fileno())
            size = fs.st_size
            rng = self.headers.get("Range", "").strip()
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
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)
        finally:
            f.close()

    def _send_json(self, obj: dict, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path != "/process":
            self.send_response(404); self.end_headers(); return

        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                self._send_json({"error": "Empty body"}, 400); return
            if n > 50 * 1024 * 1024:
                self._send_json({"error": "Too large (>50 MB)"}, 413); return
            blob = self.rfile.read(n)

            waveform, sr = decode_audio(blob)
            print(f"  Decoded {len(waveform)} samples at {sr} Hz ({len(waveform)/sr:.2f}s)")

            # Save the WAV + spectrogram into web/recordings/ so the browser can
            # fetch them via <audio src=...> for playback. Filename based on timestamp.
            stamp = int(time.time() * 1000)
            web_dir = Path(__file__).parent / "web" / "recordings"
            web_dir.mkdir(parents=True, exist_ok=True)
            wav_path = web_dir / f"rec_{stamp}.wav"
            spec_path = web_dir / f"rec_{stamp}.png"

            # Save WAV at 16 kHz mono (matches our pipeline)
            waveform_16k = waveform if sr == SAMPLE_RATE else resample_linear(waveform, sr, SAMPLE_RATE)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes((np.clip(waveform_16k, -1, 1) * 32767).astype(np.int16).tobytes())

            result = process_audio(waveform_16k, SAMPLE_RATE, spec_path)
            result["audio_file"] = f"recordings/{wav_path.name}"
            result["spectrogram_file"] = f"recordings/{spec_path.name}"
            self._send_json(result)
            print(f"  ✓ Processed in {result['model_meta']['inference_ms']} ms total")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, format, *args):
        if "POST" in format % args:
            print(f"{self.address_string()} - {format % args}")


def main():
    from nanobreath import config as cfg
    default_bh = cfg.RUNS_DIR / "v8-bce-2026-05-19" / "best.pth"

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8421)
    p.add_argument("--nanopitch", type=Path, default=cfg.NANOPITCH_CHECKPOINT,
                   help="NanoPitch backbone checkpoint. Defaults to the local "
                        "models/nanopitch/best.pth (or $NANOPITCH_CHECKPOINT).")
    p.add_argument("--breath-head", type=Path,
                   default=(default_bh if default_bh.exists() else None),
                   help="Trained BreathHead checkpoint. Defaults to the bundled "
                        "v8 release checkpoint under runs/.")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--method", choices=["peak", "threshold"], default="peak",
                   help="Event extraction algorithm (default 'peak' is robust to "
                        "under-confident model outputs).")
    p.add_argument("--phrases-from", choices=["breath_head", "ruinskiy"], default="breath_head")
    args = p.parse_args()

    if args.nanopitch is None:
        sys.exit("No NanoPitch backbone found. Place model.py + best.pth in "
                 "models/nanopitch/, or pass --nanopitch / set $NANOPITCH_CHECKPOINT.")
    if args.breath_head is None:
        sys.exit("No BreathHead checkpoint found. Pass --breath-head, or keep "
                 "runs/v8-bce-2026-05-19/best.pth in the repo.")

    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        sys.exit(f"web/ dir not found at {web_dir}")

    print(f"Loading NanoPitch from {args.nanopitch}...")
    _models["nanopitch"] = load_nanopitch(args.nanopitch)
    print(f"Loading BreathHead from {args.breath_head}...")
    _models["head"], _models["hidden"] = load_breath_head(args.breath_head)
    _models["detector"] = RuinskiyDetector()
    _models["threshold"] = args.threshold
    _models["method"] = args.method
    _models["phrases_from"] = args.phrases_from
    print(f"Models loaded ({sum(p.numel() for p in _models['head'].parameters())} BreathHead params)")

    # Serve from web/ as the document root
    import os
    os.chdir(web_dir)
    print(f"Serving {web_dir} on http://localhost:{args.port}/")
    print(f"  GET  /          serve UI")
    print(f"  POST /process   upload audio, get predictions JSON")
    httpd = ThreadingHTTPServer(("", args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
