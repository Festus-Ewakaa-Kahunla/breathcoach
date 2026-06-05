#!/usr/bin/env python3
"""
Convert Opencpop's transcriptions into our .breath.json GOLD labels.

Opencpop (Wang et al., Interspeech 2022, CC BY-NC-ND) manually annotates every
aspirate/breath as the phoneme token "AP" and every silent pause as "SP", with
per-phoneme durations. We extract the AP segments as ground-truth breath events.

transcriptions.txt format (pipe-separated, per utterance):
    name | text | phonemes | notes | note_durs | phoneme_durs | slur_flags
where `phonemes` and `phoneme_durs` are space-separated and aligned. AP = breath,
SP = silence. Cumulative phoneme_durs give each token's [start, end] in seconds.

LICENSE NOTE: Opencpop is CC BY-NC-ND. Use the output as an INTERNAL evaluation
benchmark only — do NOT commit the audio or these derived labels to a public repo
(data/ is gitignored). Cite the corpus; report metrics; do not redistribute.

USAGE
    python -m nanobreath.data.opencpop_to_breath \
        --segments-dir data/opencpop/segments \
        --out-dir data/opencpop_gold
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

BREATH_TOKEN = "AP"   # aspirate / audible breath
SILENCE_TOKEN = "SP"  # silent pause (negative class, not a breath)


def parse_line(line: str):
    """Return (name, phonemes[list], durs[list of float]) or None if malformed."""
    parts = [p.strip() for p in line.rstrip("\n").split("|")]
    if len(parts) < 6:
        return None
    name = parts[0]
    phonemes = parts[2].split()
    try:
        durs = [float(x) for x in parts[5].split()]
    except ValueError:
        return None
    if len(phonemes) != len(durs):
        return None
    return name, phonemes, durs


def breath_events(phonemes, durs):
    """AP tokens -> breath events; also return SP (silence) spans for FP stratification."""
    t = 0.0
    breaths, silences = [], []
    for ph, d in zip(phonemes, durs):
        if ph == BREATH_TOKEN:
            breaths.append((round(t, 4), round(t + d, 4)))
        elif ph == SILENCE_TOKEN:
            silences.append((round(t, 4), round(t + d, 4)))
        t += d
    return breaths, silences, t


def main():
    p = argparse.ArgumentParser(description="Opencpop AP/SP -> .breath.json gold labels")
    p.add_argument("--segments-dir", type=Path, required=True,
                   help="Opencpop segments dir (contains transcriptions.txt and wavs/)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--transcriptions", type=Path, default=None,
                   help="override path to transcriptions.txt")
    args = p.parse_args()

    trans = args.transcriptions or (args.segments_dir / "transcriptions.txt")
    wavs_dir = args.segments_dir / "wavs"
    if not trans.exists():
        raise SystemExit(f"transcriptions not found: {trans}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_clips = n_breaths = n_silence = 0
    total_dur = 0.0
    missing_wav = 0
    for line in trans.read_text(encoding="utf-8").splitlines():
        parsed = parse_line(line)
        if not parsed:
            continue
        name, phonemes, durs = parsed
        breaths, silences, dur = breath_events(phonemes, durs)

        wav_src = wavs_dir / f"{name}.wav"
        if wav_src.exists():
            link = args.out_dir / f"{name}.wav"
            if not link.exists():
                link.symlink_to(wav_src.resolve())
        else:
            missing_wav += 1

        data = {
            "audio_file": f"{name}.wav",
            "sample_rate": 16000,
            "duration_sec": round(dur, 4),
            "labeler": "opencpop_gold_AP",
            "label_date": date.today().isoformat(),
            "breath_events": [
                {"start_sec": s, "end_sec": e, "confidence": "high", "source": "opencpop_AP"}
                for s, e in breaths
            ],
            "silence_spans": [{"start_sec": s, "end_sec": e} for s, e in silences],
            "notes": "GOLD breath labels from Opencpop AP tokens (CC BY-NC-ND, internal eval only).",
        }
        (args.out_dir / f"{name}.breath.json").write_text(json.dumps(data, indent=2))
        n_clips += 1
        n_breaths += len(breaths)
        n_silence += len(silences)
        total_dur += dur

    print(f"Wrote {n_clips} gold clips to {args.out_dir}")
    print(f"  {n_breaths} breath (AP) events, {n_silence} silence (SP) spans, "
          f"{total_dur/60:.1f} min ({n_breaths/max(total_dur,1e-6)*60:.1f} breaths/min)")
    if missing_wav:
        print(f"  WARNING: {missing_wav} clips had no matching wav under {wavs_dir}")


if __name__ == "__main__":
    main()
