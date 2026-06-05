"""Seed hard-negative *candidates* into existing .breath.json files.

Runs a trained model (default v13) at a LOW threshold over the labeled clips,
finds firings that don't overlap any human-confirmed breath/uncertain marker,
and adds them to the `hard_negatives` list with source="v13_low_thresh_candidate".

The user then reviews each candidate in the labeler:
  A → confirm as hard_negative (source becomes human_accepted)
  D → reject (marker removed entirely — it was actually a real breath)
  B/U/S → recategorize (source becomes human_recategorized)

Training only uses hard_negatives whose source is human-reviewed. Unreviewed
candidates are ignored, so it's safe to leave them in the file.

Usage:
    PYTHONPATH=src .venv/bin/python dataset/tools/seed_hard_negatives.py \\
        --labels-dir dataset/labels/vocalset \\
        --audio-dir data/_sung_excerpts \\
        --nanopitch models/nanopitch/best.pth \\
        --head runs/v13-vocalset-added/best.pth \\
        --min-prominence 0.06 \\
        --tolerance-sec 0.20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanobreath.eval_events import load_head, predict_events  # noqa: E402
from nanobreath.model.joint import load_backbone_frozen  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "dataset/labels/vocalset")
    parser.add_argument("--audio-dir", type=Path, default=REPO_ROOT / "data/_sung_excerpts")
    parser.add_argument("--nanopitch", type=Path, default=REPO_ROOT / "models/nanopitch/best.pth")
    parser.add_argument("--head", type=Path, default=REPO_ROOT / "runs/v13-vocalset-added/best.pth")
    parser.add_argument("--min-prominence", type=float, default=0.06,
                        help="lower than eval (0.12) → catches the model's borderline firings")
    parser.add_argument("--tolerance-sec", type=float, default=0.20,
                        help="candidate is ignored if within this many seconds of an existing marker")
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts but don't modify files")
    args = parser.parse_args()

    device = torch.device("cpu")
    nano = load_backbone_frozen(args.nanopitch, device)
    head = load_head(args.head)
    if head._backbone_state is not None:
        nano.load_state_dict(head._backbone_state)

    label_files = sorted(args.labels_dir.glob("*.breath.json"))
    if not label_files:
        print(f"No .breath.json files in {args.labels_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Seeding hard-negative candidates across {len(label_files)} files")
    print(f"  threshold prominence: {args.min_prominence}  (eval uses 0.12)")
    print(f"  exclusion tolerance:  ±{args.tolerance_sec * 1000:.0f} ms from existing markers")
    print()

    total_predicted = 0
    total_kept = 0
    total_files_modified = 0

    for lf in label_files:
        clip_id = lf.name.replace(".breath.json", "")
        wav_path = _find_audio(args.audio_dir, clip_id)
        if wav_path is None:
            print(f"  [skip] no audio for {clip_id}")
            continue

        events = _predict_low_threshold(nano, head, wav_path, args.min_prominence)
        total_predicted += len(events)

        with open(lf) as f:
            data = json.load(f)

        existing_spans = _existing_spans(data, args.tolerance_sec)
        existing_hard_neg_starts = {
            round(ev["start_sec"], 3) for ev in data.get("hard_negatives", [])
        }

        new_candidates = []
        for ev in events:
            t_start = ev["start_sec"]
            if any(s_lo <= t_start <= s_hi for s_lo, s_hi in existing_spans):
                continue
            if round(t_start, 3) in existing_hard_neg_starts:
                continue
            new_candidates.append({
                "start_sec": round(ev["start_sec"], 3),
                "end_sec": round(ev["end_sec"], 3),
                "confidence": "low",
                "source": "v13_low_thresh_candidate",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        if not new_candidates:
            print(f"  {clip_id}: {len(events)} predicted → 0 new (all overlap existing)")
            continue

        print(f"  {clip_id}: {len(events)} predicted → {len(new_candidates)} new candidates")
        total_kept += len(new_candidates)
        total_files_modified += 1

        if args.dry_run:
            continue

        data.setdefault("hard_negatives", []).extend(new_candidates)
        data["last_modified"] = datetime.now(timezone.utc).isoformat()
        with open(lf, "w") as f:
            json.dump(data, f, indent=2)

    print()
    print(f"Total predictions:       {total_predicted}")
    print(f"Total kept as candidates: {total_kept}")
    print(f"Files modified:           {total_files_modified} / {len(label_files)}")
    if args.dry_run:
        print("(dry-run — no files written)")


def _existing_spans(data: dict, tol_sec: float) -> list[tuple[float, float]]:
    """Build padded (start, end) spans for all existing markers so candidates
    that fall inside any span are excluded."""
    spans: list[tuple[float, float]] = []
    for key in ("breath_events", "silent_breaths", "uncertain", "exhales"):
        for ev in data.get(key, []):
            spans.append((ev["start_sec"] - tol_sec, ev["end_sec"] + tol_sec))
    return spans


def _find_audio(audio_dir: Path, clip_id: str) -> Path | None:
    candidates = list(audio_dir.rglob(f"{clip_id}.wav"))
    return candidates[0] if candidates else None


def _predict_low_threshold(nano, head, wav_path: Path, min_prominence: float) -> list[dict]:
    """Run the model and pull peaks at a lower prominence than eval uses."""
    from nanobreath.data.dataset import (
        load_labeled_clip, compute_log_mel, compute_breath_features, N_EXTRA_FEATURES,
    )
    from nanobreath.model.joint import JointModel
    from nanobreath.deployment.precompute_predictions import peak_events, apply_calibration

    clip = load_labeled_clip(wav_path)
    mel = compute_log_mel(clip.waveform)
    joint = JointModel(nano, head)
    joint.eval()
    extra_t = None
    if getattr(head, "_use_extra", False):
        feats = compute_breath_features(clip.waveform, mel)
        extra_t = torch.from_numpy(feats).unsqueeze(0)
    with torch.no_grad():
        _vad, _pitch, breath = joint(torch.from_numpy(mel).unsqueeze(0), extra_features=extra_t)
    bp = apply_calibration(breath.squeeze().numpy(), getattr(head, "_calibration", None))
    return peak_events(bp, min_prominence=min_prominence, min_distance_sec=1.0)


if __name__ == "__main__":
    main()
