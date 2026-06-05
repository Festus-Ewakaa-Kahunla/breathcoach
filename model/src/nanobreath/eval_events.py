#!/usr/bin/env python3
"""
Event-based breath evaluation against GOLD labels, with false-positives stratified
by what they fired on (silence vs voiced). This is the metric the literature
prescribes for "does it fire on real inhales, not on singing/silence":

  - Event F1 at an onset tolerance (DCASE sed_eval / MIREX convention; ±200 ms
    primary for soft breath onsets, ±100 ms as a stricter check).
  - Predicted events are matched one-to-one to gold breath (AP) onsets.
  - Every FALSE POSITIVE is classified: did it land on a gold SILENCE (SP) span
    (-> "silence FP") or elsewhere, i.e. voiced singing (-> "voiced FP")?

Gold labels come from opencpop_to_breath.py (Opencpop AP=breath, SP=silence).

USAGE
    python -m nanobreath.eval_events \
        --gold-dir data/opencpop_gold \
        --nanopitch models/nanopitch/best.pth \
        --checkpoints runs/v8-bce-2026-05-19/best.pth runs/v9-respiro/best.pth
"""
from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
import torch

torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.PurePosixPath])

from nanobreath.model.breath_head import BreathHead
from nanobreath.model.joint import JointModel, load_backbone_frozen
from nanobreath.data.dataset import (
    load_labeled_clip, compute_log_mel, compute_breath_features, N_EXTRA_FEATURES,
)
from nanobreath.deployment.precompute_predictions import peak_events, apply_calibration


def load_head(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    saved_args = ck.get("args", {})
    hidden = saved_args.get("hidden", 8)
    use_extra = saved_args.get("use_extra_features", False)
    in_features = 384 + (N_EXTRA_FEATURES if use_extra else 0)
    head = BreathHead(in_features=in_features, hidden=hidden, kernel_size=5)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    head._calibration = ck.get("calibration")
    head._backbone_state = ck.get("backbone_state_dict")  # set when --finetune-backbone was used
    head._use_extra = use_extra
    return head


def predict_events(nano, head, wav_path: Path, voicing_veto: bool = False):
    clip = load_labeled_clip(wav_path)
    mel = compute_log_mel(clip.waveform)
    joint = JointModel(nano, head)
    joint.eval()
    extra_t = None
    if getattr(head, "_use_extra", False):
        feats = compute_breath_features(clip.waveform, mel)
        extra_t = torch.from_numpy(feats).unsqueeze(0)
    with torch.no_grad():
        vad, _pitch, breath = joint(torch.from_numpy(mel).unsqueeze(0), extra_features=extra_t)
    bp = apply_calibration(breath.squeeze().numpy(), getattr(head, "_calibration", None))
    if voicing_veto:
        v = vad.squeeze().numpy()
        bp = np.where(v > 0.5, 0.0, bp)
    return peak_events(bp, min_prominence=0.12, min_distance_sec=1.0)


def match_events(pred, gold, tol):
    """Greedy one-to-one onset matching. Returns (tp, fp_list, fn) — fp_list are
    unmatched predicted events (for stratification)."""
    gold_on = sorted(g["start_sec"] for g in gold)
    used = [False] * len(gold_on)
    tp = 0
    fp_list = []
    for ev in sorted(pred, key=lambda e: e["start_sec"]):
        t = ev["start_sec"]
        best, best_d = -1, tol + 1e-9
        for i, go in enumerate(gold_on):
            if used[i]:
                continue
            d = abs(t - go)
            if d <= best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            tp += 1
        else:
            fp_list.append(ev)
    fn = used.count(False)
    return tp, fp_list, fn


def in_any_span(t, spans):
    return any(s["start_sec"] <= t <= s["end_sec"] for s in spans)


def evaluate(nano, head, gold_files, tolerances, voicing_veto: bool = False):
    per_tol = {tol: {"tp": 0, "fp": 0, "fn": 0} for tol in tolerances}
    fp_silence = fp_voiced = 0
    n_pred = 0
    total_min = 0.0
    for jf in gold_files:
        gold = json.loads(jf.read_text())
        breaths = gold["breath_events"]
        silences = gold.get("silence_spans", [])
        wav = jf.with_name(jf.name.replace(".breath.json", ".wav"))
        if not wav.exists():
            continue
        pred = predict_events(nano, head, wav, voicing_veto=voicing_veto)
        n_pred += len(pred)
        total_min += gold["duration_sec"] / 60.0
        for tol in tolerances:
            tp, fp_list, fn = match_events(pred, breaths, tol)
            per_tol[tol]["tp"] += tp
            per_tol[tol]["fp"] += len(fp_list)
            per_tol[tol]["fn"] += fn
            if tol == max(tolerances):  # stratify FPs at the loosest tolerance
                for ev in fp_list:
                    if in_any_span(ev["start_sec"], silences):
                        fp_silence += 1
                    else:
                        fp_voiced += 1
    out = {"n_pred": n_pred, "rate": n_pred / max(total_min, 1e-9),
           "fp_silence": fp_silence, "fp_voiced": fp_voiced, "per_tol": {}}
    for tol, c in per_tol.items():
        p = c["tp"] / max(c["tp"] + c["fp"], 1)
        r = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        out["per_tol"][tol] = {"P": p, "R": r, "F1": f1, **c}
    return out


def main():
    ap = argparse.ArgumentParser(description="Event-based breath eval with stratified FPs")
    ap.add_argument("--gold-dir", type=Path, required=True)
    ap.add_argument("--nanopitch", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.2, 0.1])
    ap.add_argument("--voicing-veto", action="store_true",
                    help="post-processing: zero out breath_prob where NanoPitch VAD > 0.5")
    args = ap.parse_args()

    gold_files = sorted(args.gold_dir.glob("*.breath.json"))
    if not gold_files:
        raise SystemExit(f"No gold .breath.json in {args.gold_dir}")
    n_gold = sum(len(json.loads(f.read_text())["breath_events"]) for f in gold_files)
    print(f"Gold: {len(gold_files)} clips, {n_gold} breath events\n")

    nano = load_backbone_frozen(args.nanopitch, torch.device("cpu"))
    orig_backbone_sd = {k: v.detach().clone() for k, v in nano.state_dict().items()}
    for ckpt in args.checkpoints:
        head = load_head(ckpt)
        # Swap in this checkpoint's finetuned backbone if it has one; else restore original.
        if head._backbone_state is not None:
            nano.load_state_dict(head._backbone_state)
            print(f"  (using finetuned backbone shipped with {ckpt.name})")
        else:
            nano.load_state_dict(orig_backbone_sd)
        r = evaluate(nano, head, gold_files, sorted(args.tolerances, reverse=True),
                     voicing_veto=args.voicing_veto)
        tag = " + voicing-veto" if args.voicing_veto else ""
        print(f"=== {ckpt.parent.name}/{ckpt.name}{tag} ===")
        print(f"  predicted {r['n_pred']} events ({r['rate']:.1f}/min)")
        for tol in sorted(args.tolerances):
            m = r["per_tol"][tol]
            print(f"  @±{int(tol*1000)}ms: F1={m['F1']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        tot_fp = r["fp_silence"] + r["fp_voiced"]
        if tot_fp:
            print(f"  false positives landed on:  silence(SP) {r['fp_silence']} "
                  f"({100*r['fp_silence']/tot_fp:.0f}%)   voiced-singing {r['fp_voiced']} "
                  f"({100*r['fp_voiced']/tot_fp:.0f}%)")
        print()


if __name__ == "__main__":
    main()
