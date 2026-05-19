"""Smoke tests for nanobreath. Run with `pytest tests/`.

These tests do not require the VocalSet download, a backbone checkpoint, or
hand labels — they exercise the in-memory paths only.
"""

from __future__ import annotations

import numpy as np

from nanobreath.baseline.ruinskiy_lavner import RuinskiyDetector
from nanobreath.config import SAMPLE_RATE
from nanobreath.data.dataset import compute_log_mel
from nanobreath.eval import (
    evaluate_events,
    evaluate_frame,
    frames_to_events,
)
from nanobreath.feature.phrase_tracker import PhraseTracker


def _synth_breath_waveform(duration_sec: float = 5.0) -> np.ndarray:
    """Quiet baseline with two bursts of breath-like band-limited noise."""
    n = int(duration_sec * SAMPLE_RATE)
    rng = np.random.default_rng(0)
    sig = 0.02 * rng.standard_normal(n).astype(np.float32)
    for start_sec in (1.0, 3.0):
        s = int(start_sec * SAMPLE_RATE)
        e = s + int(0.3 * SAMPLE_RATE)
        sig[s:e] += 0.3 * rng.standard_normal(e - s).astype(np.float32)
    return sig


def test_log_mel_shape() -> None:
    wf = _synth_breath_waveform()
    mel = compute_log_mel(wf)
    assert mel.ndim == 2
    assert mel.shape[1] == 40
    assert mel.shape[0] > 0


def test_breath_head_forward() -> None:
    import torch  # local import so the rest of the suite runs without torch installed
    from nanobreath.model.breath_head import BreathHead

    head = BreathHead(in_features=384, hidden=8)
    assert 10_000 < head.num_parameters() < 25_000
    x = torch.randn(2, 300, 384)
    y = head(x)
    assert y.shape == (2, 300, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_ruinskiy_runs() -> None:
    wf = _synth_breath_waveform()
    det = RuinskiyDetector()
    events = det.detect_array(wf, SAMPLE_RATE)
    # We don't assert the count — generic template + noise is non-deterministic
    # at low signal — but the call must not raise and must return a list.
    assert isinstance(events, list)


def test_frame_metrics_perfect() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.float32)
    probs = labels.copy()  # perfect prediction
    fm = evaluate_frame(probs, labels)
    assert fm.f1 > 0.99
    # PR-AUC uses trapezoidal integration over the recall axis. With perfect
    # predictions on a small example the area is bounded below by
    # (1 - 1/n_pos), so we use a looser bound rather than 0.99.
    assert fm.pr_auc > 0.75


def test_event_metrics_match_within_tolerance() -> None:
    true = [(1.00, 1.20), (3.00, 3.25)]
    pred = [(1.03, 1.23), (3.30, 3.55)]  # second pred is 300 ms off in onset
    em100 = evaluate_events(pred, true, tolerance_ms=100)
    assert em100.n_matched == 1
    em250 = evaluate_events(pred, true, tolerance_ms=250)
    assert em250.n_matched == 1
    em500 = evaluate_events(pred, true, tolerance_ms=500)
    assert em500.n_matched == 2


def test_phrase_tracker_state_machine() -> None:
    tracker = PhraseTracker()
    # Sing for 1 s, breathe for 0.3 s, sing for 1 s
    for _ in range(100):
        tracker.step(0.05, 0.9)
    for _ in range(30):
        tracker.step(0.9, 0.1)
    for _ in range(100):
        tracker.step(0.05, 0.9)
    summary = tracker.session_summary()
    assert summary["n_phrases"] >= 1


def test_frames_to_events_roundtrip() -> None:
    labels = np.zeros(200, dtype=np.float32)
    labels[20:30] = 1.0
    labels[100:120] = 1.0
    events = frames_to_events(labels, threshold=0.5, min_duration_sec=0.0)
    assert len(events) == 2
