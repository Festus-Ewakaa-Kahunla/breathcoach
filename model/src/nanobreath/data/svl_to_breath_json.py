#!/usr/bin/env python3
"""
Convert Sonic Visualizer .svl annotation files to our canonical .breath.json format.

WHY THIS EXISTS

Sonic Visualizer saves annotations as .svl (a custom XML format). Our training
pipeline expects .breath.json with (start_sec, end_sec) tuples per breath event.
This script bridges the two so labeling work in Sonic Visualizer can be consumed
directly by train.py without manual editing.

WHAT .SVL LOOKS LIKE

A Time Instants layer (the recommended workflow — single time marker per breath):

    <sv>
      <data>
        <model id="1" name="Breath events" sampleRate="16000" type="sparse"
               dimensions="1" resolution="1" notifyOnAdd="true"
               dataset="0" subtype="point" />
        <dataset id="0" dimensions="1">
          <point frame="29320" label="" />
          <point frame="75400" label="" />
          <point frame="146880" label="" />
        </dataset>
      </data>
    </sv>

Each <point frame="N"/> is one breath onset. We convert N / sampleRate → start_sec.
Since instants have no duration, we assign end_sec = start_sec + DEFAULT_DURATION
(spec: label_format.md says 0.25 s when only starts are marked).

A Time Values layer (start + duration via "value" field as end-time, or via
"duration" attribute) is also supported — we read duration if present, otherwise
use the default.

USAGE

    python svl_to_breath_json.py path/to/f1_dona_vibrato.svl
        → writes f1_dona_vibrato.breath.json next to the .svl

    python svl_to_breath_json.py path/to/labels/ --recursive
        → processes every .svl in the directory, pairs each with the WAV of
          the same stem, writes .breath.json siblings.

    python svl_to_breath_json.py file.svl --audio path/to/file.wav --labeler Festus
        → explicit audio path + labeler name for metadata.

DESIGN NOTES

- Frame numbers in .svl are in SAMPLES (not in 10-ms model frames). We divide
  by sampleRate (read from the .svl model attribute) to get seconds.
- If multiple <model> blocks exist, we use the FIRST sparse/point model.
  Add a --model-name flag if you start using multiple layers.
- Audio duration is read from the WAV if available (lets the training loader
  validate that labels don't exceed audio length). If the WAV isn't found
  we still write the JSON but with duration_sec=null and a warning.
- DEFAULT_END_OFFSET (0.25 s) matches label_format.md line 61.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Optional


DEFAULT_END_OFFSET_SEC = 0.25  # used when .svl only has start instants


def parse_svl(svl_path: Path) -> tuple[list[dict], int]:
    """Parse an .svl file and return (events, sample_rate).

    Each event is a dict: {start_sec, end_sec, confidence, source}.
    """
    tree = ET.parse(svl_path)
    root = tree.getroot()

    # Find the first sparse point model — that's our breath layer.
    # Sonic Visualizer .svl files have <data><model .../><dataset>...</dataset></data>
    # We don't want region/instant/value distinction to trip us up; we look at
    # whichever <model> has type="sparse" and read its <dataset>.
    model = None
    for m in root.iter("model"):
        if m.attrib.get("type") == "sparse":
            model = m
            break
    if model is None:
        raise ValueError(f"No sparse model found in {svl_path}. "
                         f"Did you add a Time Instants Layer in Sonic Visualizer?")

    sample_rate = int(model.attrib.get("sampleRate", "16000"))
    dataset_id = model.attrib.get("dataset")

    # Find the matching dataset by id
    dataset = None
    for d in root.iter("dataset"):
        if d.attrib.get("id") == dataset_id:
            dataset = d
            break
    if dataset is None:
        raise ValueError(f"No dataset with id={dataset_id} in {svl_path}")

    events = []
    for point in dataset.iter("point"):
        frame = int(point.attrib["frame"])
        start_sec = frame / sample_rate

        # If the point has a duration attribute (regions / time-values layer),
        # use it. Otherwise fall back to DEFAULT_END_OFFSET_SEC.
        duration_attr = point.attrib.get("duration")
        if duration_attr is not None:
            duration_sec = int(duration_attr) / sample_rate
            end_sec = start_sec + duration_sec
            source_note = "duration from .svl"
        else:
            end_sec = start_sec + DEFAULT_END_OFFSET_SEC
            source_note = f"end_sec auto-set to start_sec + {DEFAULT_END_OFFSET_SEC}s"

        # Label text — Sonic Visualizer lets you type labels per point.
        # We treat empty label as "high confidence", any non-empty as a note.
        label_text = (point.attrib.get("label") or "").strip()
        confidence = "low" if label_text.lower() in {"?", "maybe", "low"} else "high"

        events.append({
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "confidence": confidence,
            "source": source_note,
            **({"label_text": label_text} if label_text else {}),
        })

    # Sort by start time (Sonic Visualizer doesn't guarantee order)
    events.sort(key=lambda e: e["start_sec"])
    return events, sample_rate


def get_audio_duration(wav_path: Path) -> Optional[float]:
    """Return duration in seconds, or None if the WAV is missing / unreadable."""
    if not wav_path.exists():
        warnings.warn(f"WAV not found: {wav_path} — duration_sec will be null")
        return None
    try:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
        return round(n_frames / rate, 4)
    except Exception as exc:
        warnings.warn(f"Could not read {wav_path}: {exc}")
        return None


def find_audio_for_svl(svl_path: Path) -> Optional[Path]:
    """Find a WAV next to the .svl with the same stem."""
    candidate = svl_path.with_suffix(".wav")
    if candidate.exists():
        return candidate
    # Sometimes Sonic Visualizer saves .svl with a slightly different stem;
    # try the directory for any wav matching the prefix.
    parent = svl_path.parent
    stem = svl_path.stem
    matches = list(parent.glob(f"{stem}*.wav"))
    return matches[0] if matches else None


def convert_one(svl_path: Path,
                audio_path: Optional[Path] = None,
                labeler: str = "Festus",
                notes: str = "") -> Path:
    """Convert one .svl → .breath.json. Returns path to the written JSON."""
    events, sample_rate = parse_svl(svl_path)

    if audio_path is None:
        audio_path = find_audio_for_svl(svl_path)
    duration_sec = get_audio_duration(audio_path) if audio_path else None

    out = {
        "audio_file": audio_path.name if audio_path else None,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "labeler": labeler,
        "label_date": date.today().isoformat(),
        "breath_events": events,
        "notes": notes,
        "source_svl": svl_path.name,
    }

    json_path = svl_path.with_suffix(".breath.json")
    json_path.write_text(json.dumps(out, indent=2))
    return json_path


def main():
    p = argparse.ArgumentParser(description="Convert Sonic Visualizer .svl → .breath.json")
    p.add_argument("path", type=Path,
                   help="path to an .svl file, OR a directory if --recursive")
    p.add_argument("--audio", type=Path,
                   help="path to the WAV (defaults to same-stem .wav next to .svl)")
    p.add_argument("--labeler", default="Festus")
    p.add_argument("--notes", default="")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="if path is a directory, convert every .svl inside it")
    args = p.parse_args()

    if args.path.is_dir():
        if not args.recursive:
            p.error(f"{args.path} is a directory; pass --recursive to process all .svl in it")
        svl_files = sorted(args.path.rglob("*.svl"))
        if not svl_files:
            print(f"No .svl files under {args.path}", file=sys.stderr)
            sys.exit(1)
        print(f"Converting {len(svl_files)} .svl files...")
        for svl in svl_files:
            try:
                out = convert_one(svl, labeler=args.labeler, notes=args.notes)
                n_events = len(json.loads(out.read_text())["breath_events"])
                print(f"  ✓ {svl.name} → {out.name} ({n_events} breath events)")
            except Exception as exc:
                print(f"  ✗ {svl.name}: {exc}", file=sys.stderr)
    else:
        out = convert_one(args.path,
                          audio_path=args.audio,
                          labeler=args.labeler,
                          notes=args.notes)
        n_events = len(json.loads(out.read_text())["breath_events"])
        print(f"Wrote {out} ({n_events} breath events)")


if __name__ == "__main__":
    main()
