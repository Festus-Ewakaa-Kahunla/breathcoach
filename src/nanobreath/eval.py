"""
Evaluation metrics for breath-event detection.

Reports both:
- Frame-level metrics (precision/recall/F1, PR-AUC)
- Event-level metrics with tolerance windows (50, 100, 250 ms)

Frame-level captures whether the model correctly identifies each 10ms frame
as breath or not. PR-AUC matters most because the class is highly imbalanced
(~5% breath frames).

Event-level captures whether breath events are correctly localized as discrete
units, which is what the user-facing UI actually needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


HOP_SEC = 0.010  # 10 ms — matches NanoPitch frame rate


@dataclass
class FrameMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    threshold_at_best_f1: float
    n_positive_frames: int
    n_total_frames: int

    def __str__(self) -> str:
        return (f"Frame  P={self.precision:.3f}  R={self.recall:.3f}  "
                f"F1={self.f1:.3f}  PR-AUC={self.pr_auc:.3f}  "
                f"(best @ thresh={self.threshold_at_best_f1:.2f}, "
                f"{self.n_positive_frames}/{self.n_total_frames} positive)")


@dataclass
class EventMetrics:
    tolerance_ms: int
    precision: float
    recall: float
    f1: float
    n_pred_events: int
    n_true_events: int
    n_matched: int

    def __str__(self) -> str:
        return (f"Event@{self.tolerance_ms}ms  P={self.precision:.3f}  "
                f"R={self.recall:.3f}  F1={self.f1:.3f}  "
                f"({self.n_matched}/{self.n_true_events} matched, "
                f"{self.n_pred_events} predicted)")


def evaluate_frame(probs: np.ndarray, labels: np.ndarray,
                   threshold_grid: np.ndarray | None = None) -> FrameMetrics:
    """Frame-level metrics from per-frame predictions and binary labels.

    Args:
        probs:  (n_frames,) float predictions in [0, 1]
        labels: (n_frames,) binary {0, 1}
        threshold_grid: thresholds to sweep for best-F1; default 0.05 ... 0.95

    Returns:
        FrameMetrics with precision/recall/F1 at the best-F1 threshold and PR-AUC.
    """
    probs = np.asarray(probs).ravel()
    labels = np.asarray(labels).ravel().astype(np.int32)
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 1.0, 0.05)

    n_pos = int(labels.sum())
    n_tot = len(labels)
    if n_pos == 0:
        return FrameMetrics(0.0, 0.0, 0.0, 0.0, 0.5, 0, n_tot)

    # PR-AUC via ranking
    pr_auc = _pr_auc(probs, labels)

    # Sweep for best F1
    best_f1 = -1.0
    best_p, best_r, best_t = 0.0, 0.0, 0.5
    for t in threshold_grid:
        preds = (probs >= t).astype(np.int32)
        tp = int((preds * labels).sum())
        fp = int((preds * (1 - labels)).sum())
        fn = int(((1 - preds) * labels).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r / max(1e-10, p + r))
        if f1 > best_f1:
            best_f1, best_p, best_r, best_t = f1, p, r, float(t)

    return FrameMetrics(best_p, best_r, best_f1, pr_auc, best_t, n_pos, n_tot)


def evaluate_events(pred_events: List[Tuple[float, float]],
                    true_events: List[Tuple[float, float]],
                    tolerance_ms: int) -> EventMetrics:
    """Event-level metrics with onset-tolerance matching.

    Each predicted event is matched to a true event if their onsets are within
    `tolerance_ms` of each other. Greedy matching: closest-first.

    Args:
        pred_events: list of (start_sec, end_sec) for predicted breath events
        true_events: list of (start_sec, end_sec) for ground-truth breath events
        tolerance_ms: matching tolerance on onset (in ms)

    Returns:
        EventMetrics with precision, recall, F1.
    """
    tol = tolerance_ms / 1000.0
    used_true = [False] * len(true_events)
    matched = 0

    # Greedy: for each prediction, find the closest unused true event within tolerance
    for p_start, _p_end in pred_events:
        best_idx = -1
        best_dist = float("inf")
        for j, (t_start, _t_end) in enumerate(true_events):
            if used_true[j]:
                continue
            dist = abs(p_start - t_start)
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx >= 0:
            used_true[best_idx] = True
            matched += 1

    n_pred = len(pred_events)
    n_true = len(true_events)
    p = matched / max(1, n_pred)
    r = matched / max(1, n_true)
    f1 = 2 * p * r / max(1e-10, p + r)

    return EventMetrics(
        tolerance_ms=tolerance_ms,
        precision=p, recall=r, f1=f1,
        n_pred_events=n_pred,
        n_true_events=n_true,
        n_matched=matched,
    )


def frames_to_events(frame_probs: np.ndarray, threshold: float = 0.5,
                     min_duration_sec: float = 0.05,
                     hop_sec: float = HOP_SEC) -> List[Tuple[float, float]]:
    """Threshold per-frame probabilities and group into discrete events.

    Drops events shorter than min_duration_sec.
    """
    binary = (frame_probs >= threshold).astype(np.int32)
    events = []
    in_run = False
    run_start = 0
    for t in range(len(binary)):
        if binary[t] and not in_run:
            in_run, run_start = True, t
        elif not binary[t] and in_run:
            in_run = False
            dur = (t - run_start) * hop_sec
            if dur >= min_duration_sec:
                events.append((run_start * hop_sec, t * hop_sec))
    if in_run:
        dur = (len(binary) - run_start) * hop_sec
        if dur >= min_duration_sec:
            events.append((run_start * hop_sec, len(binary) * hop_sec))
    return events


def labels_to_events(label_array: np.ndarray, hop_sec: float = HOP_SEC) -> List[Tuple[float, float]]:
    """Convert a binary frame-level label array into a list of (start_sec, end_sec)."""
    return frames_to_events(label_array.astype(np.float32),
                            threshold=0.5, min_duration_sec=0.0, hop_sec=hop_sec)


def _pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal PR-AUC. No external deps."""
    order = np.argsort(-probs)
    labels = labels[order]
    cumsum_pos = np.cumsum(labels)
    cumsum_total = np.arange(1, len(labels) + 1)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    precision = cumsum_pos / cumsum_total
    recall = cumsum_pos / n_pos
    # Trapezoidal AUC over recall axis
    return float(np.trapz(precision, recall))


def evaluate_clip(probs: np.ndarray, labels: np.ndarray,
                  pred_threshold: float = 0.5) -> Dict[str, object]:
    """Run full evaluation on one clip's predictions vs ground-truth labels.

    Returns a dict with both frame and event metrics.
    """
    frame_metrics = evaluate_frame(probs, labels)
    pred_events = frames_to_events(probs, threshold=frame_metrics.threshold_at_best_f1)
    true_events = labels_to_events(labels)
    event_metrics = {
        ms: evaluate_events(pred_events, true_events, tolerance_ms=ms)
        for ms in (50, 100, 250)
    }
    return {
        "frame": frame_metrics,
        "event": event_metrics,
        "pred_events": pred_events,
        "true_events": true_events,
    }
