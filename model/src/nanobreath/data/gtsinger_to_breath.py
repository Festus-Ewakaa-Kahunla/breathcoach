#!/usr/bin/env python3
"""
Convert GTSinger's word-level JSON annotations into our .breath.json labels.

GTSinger (Zhang et al., NeurIPS 2024, CC BY-NC-SA 4.0) — the corpus NanoPitch was
trained on — annotates each audible breath as the word token "<AP>" (aspirate) and
each silent pause as "<SP>", with start/end times. We extract the <AP> spans as
breath events (and <SP> as silence, for false-positive stratification).

Released JSON is a list of word entries:
    [{"word": "and", "start_time": 0.0, "end_time": 0.17, "ph": [...], ...},
     {"word": "<AP>", "start_time": 4.2, "end_time": 4.6, ...}, ...]

These are REAL (human-corrected, MFA-aligned) labels — unlike the Ruinskiy/Respiro
pseudo-labels — so they serve as both a gold eval set and a real training target.

USAGE
    python -m nanobreath.data.gtsinger_to_breath /tmp/gts_en \
        --clips-list /tmp/test_clips.txt --out-dir data/gtsinger_gold_test
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

BREATH = "<AP>"
SILENCE = "<SP>"


def convert_clip(json_path: Path, wav_path: Path, out_dir: Path) -> tuple[int, int]:
    entries = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        return 0, 0
    breaths = [(e["start_time"], e["end_time"]) for e in entries if e.get("word") == BREATH]
    silences = [(e["start_time"], e["end_time"]) for e in entries if e.get("word") == SILENCE]
    duration = max((e.get("end_time", 0.0) for e in entries), default=0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Unique flat name (paths nest by singer/technique/song/group) to avoid collisions.
    stem = "__".join(json_path.with_suffix("").parts[-4:])
    link = out_dir / f"{stem}.wav"
    if wav_path.exists() and not link.exists():
        link.symlink_to(wav_path.resolve())

    data = {
        "audio_file": f"{stem}.wav",
        "sample_rate": 16000,
        "duration_sec": round(duration, 4),
        "labeler": "gtsinger_gold_AP",
        "label_date": date.today().isoformat(),
        "breath_events": [
            {"start_sec": round(s, 4), "end_sec": round(e, 4), "confidence": "high",
             "source": "gtsinger_AP"} for s, e in breaths
        ],
        "silence_spans": [{"start_sec": round(s, 4), "end_sec": round(e, 4)} for s, e in silences],
        "notes": "GOLD breath labels from GTSinger <AP> tokens (CC BY-NC-SA 4.0).",
    }
    (out_dir / f"{stem}.breath.json").write_text(json.dumps(data, indent=2))
    return len(breaths), len(silences)


def main():
    ap = argparse.ArgumentParser(description="GTSinger <AP>/<SP> -> .breath.json")
    ap.add_argument("root", type=Path, help="GTSinger local download root")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clips-list", type=Path, default=None,
                    help="file of clip stems (relative to root, no extension); "
                         "if omitted, process every *.json under root")
    args = ap.parse_args()

    if args.clips_list:
        stems = [s.strip() for s in args.clips_list.read_text().splitlines() if s.strip()]
        jsons = [args.root / f"{s}.json" for s in stems]
    else:
        jsons = [Path(p) for p in glob.glob(str(args.root / "**" / "*.json"), recursive=True)
                 if "metadata" not in p]

    n_clips = n_breath = n_sil = missing = 0
    for jp in jsons:
        wav = jp.with_suffix(".wav")
        if not jp.exists():
            missing += 1
            continue
        b, s = convert_clip(jp, wav, args.out_dir)
        n_clips += 1
        n_breath += b
        n_sil += s
        if not wav.exists():
            missing += 1
    print(f"Wrote {n_clips} gold clips to {args.out_dir}")
    print(f"  {n_breath} breath (<AP>) events, {n_sil} silence (<SP>) spans")
    if missing:
        print(f"  WARNING: {missing} clips missing json or wav")


if __name__ == "__main__":
    main()
