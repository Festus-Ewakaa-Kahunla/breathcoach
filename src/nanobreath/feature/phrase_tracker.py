"""
Real-time phrase tracker — turns frame-level breath probabilities into
human-meaningful coaching feedback.

Inputs (per frame):
- breath_prob: float in [0, 1] from BreathHead
- voiced_prob: float in [0, 1] from NanoPitch's VAD head

Outputs (live, per call):
- current_phrase_length_sec: how long we've been singing without breathing
- last_phrase_length_sec:    duration of the most recent completed phrase
- breath_just_happened:      bool — true on the frame a breath ended
- phrase_just_ended:         bool — true on the frame a new breath starts
- coaching_message:          string — short feedback like "Nice phrase!" or "Breathe earlier next time"

Why this is a separate module:
- Browser UI consumes these high-level signals, not raw probabilities
- Coaching logic is application-specific and easier to tune in pure Python
- Same logic ports identically to JS for in-browser execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


HOP_SEC = 0.010  # NanoPitch frame rate


# Pedagogy targets — these come from voice-science literature
# (e.g. typical sustained-phrase length for trained singers ≈ 6-12 sec).
# Stay descriptive rather than prescriptive: we ENCOURAGE healthy ranges,
# we don't punish anything outside them.
PHRASE_RANGES = {
    "very_short": (0.0, 2.0),    # gasping / panicking
    "short":      (2.0, 4.0),    # OK for beginners
    "comfortable":(4.0, 8.0),    # good
    "long":       (8.0, 12.0),   # solid technique
    "very_long":  (12.0, 30.0),  # advanced
    "running_out":(30.0, 1e9),   # likely strained
}


@dataclass
class PhraseEvent:
    """One completed phrase, recorded after a breath ends it."""
    phrase_start_sec: float
    phrase_end_sec: float
    duration_sec: float


@dataclass
class TrackerState:
    """Running state for the phrase tracker."""
    # Hysteresis thresholds — once breath_prob crosses high, we say "in breath";
    # we only leave "in breath" when prob drops below low.
    high_threshold: float = 0.6
    low_threshold: float = 0.3
    # Voicing threshold — phrase only counts when the singer is actually phonating
    voiced_threshold: float = 0.5
    # Min duration for a breath to count as a real event
    min_breath_dur_sec: float = 0.05

    # Internal state
    current_time_sec: float = 0.0
    in_breath: bool = False
    breath_run_start_sec: float = 0.0
    in_phrase: bool = False
    phrase_start_sec: float = 0.0
    phrases: list = field(default_factory=list)
    last_phrase_length_sec: float = 0.0


@dataclass
class TrackerOutput:
    """One frame's worth of high-level UI signals."""
    current_phrase_length_sec: float
    last_phrase_length_sec: float
    breath_just_happened: bool
    phrase_just_ended: bool
    in_phrase: bool
    in_breath: bool
    coaching_message: str
    coaching_severity: str  # "info" | "encourage" | "warn"


class PhraseTracker:
    """Stateful phrase tracker. Call .step() once per audio frame (10 ms).

    Usage:
        tracker = PhraseTracker()
        for frame_t in range(n_frames):
            out = tracker.step(breath_prob[frame_t], voiced_prob[frame_t])
            ui.render(out)
    """

    def __init__(self,
                 high_threshold: float = 0.6,
                 low_threshold: float = 0.3,
                 voiced_threshold: float = 0.5,
                 min_breath_dur_sec: float = 0.05):
        self.s = TrackerState(
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            voiced_threshold=voiced_threshold,
            min_breath_dur_sec=min_breath_dur_sec,
        )

    def step(self, breath_prob: float, voiced_prob: float,
             hop_sec: float = HOP_SEC) -> TrackerOutput:
        """Advance one frame and return current state for the UI."""
        s = self.s
        s.current_time_sec += hop_sec

        breath_just_happened = False
        phrase_just_ended = False

        # ── Breath state machine ──
        if s.in_breath:
            if breath_prob < s.low_threshold:
                # Breath ended
                breath_dur = s.current_time_sec - s.breath_run_start_sec
                if breath_dur >= s.min_breath_dur_sec:
                    breath_just_happened = True
                s.in_breath = False
        else:
            if breath_prob > s.high_threshold:
                s.in_breath = True
                s.breath_run_start_sec = s.current_time_sec
                # If we were in a phrase, the phrase just ended
                if s.in_phrase:
                    phrase_just_ended = True
                    phrase_dur = s.current_time_sec - s.phrase_start_sec
                    s.phrases.append(PhraseEvent(
                        phrase_start_sec=s.phrase_start_sec,
                        phrase_end_sec=s.current_time_sec,
                        duration_sec=phrase_dur,
                    ))
                    s.last_phrase_length_sec = phrase_dur
                    s.in_phrase = False

        # ── Phrase state machine ──
        # A phrase starts when we are voiced AND not in a breath
        if not s.in_phrase and not s.in_breath and voiced_prob > s.voiced_threshold:
            s.in_phrase = True
            s.phrase_start_sec = s.current_time_sec

        # If voicing stops without a breath (e.g., silence between songs),
        # we don't count that as a phrase end — the user is just resting.
        # The UI can render in_phrase=False to show "no current phrase."

        current_phrase_length_sec = (
            s.current_time_sec - s.phrase_start_sec if s.in_phrase else 0.0
        )

        msg, severity = self._coaching_message(
            current_phrase_length_sec,
            s.last_phrase_length_sec,
            phrase_just_ended,
            breath_just_happened,
            s.in_phrase,
        )

        return TrackerOutput(
            current_phrase_length_sec=current_phrase_length_sec,
            last_phrase_length_sec=s.last_phrase_length_sec,
            breath_just_happened=breath_just_happened,
            phrase_just_ended=phrase_just_ended,
            in_phrase=s.in_phrase,
            in_breath=s.in_breath,
            coaching_message=msg,
            coaching_severity=severity,
        )

    # ── Coaching language ──

    def _coaching_message(self, current_len_sec: float, last_len_sec: float,
                          phrase_ended: bool, breath_happened: bool,
                          in_phrase: bool) -> tuple:
        """Pick a short, descriptive feedback string and severity."""
        if phrase_ended:
            cat = self._categorize(last_len_sec)
            if cat == "very_short":
                return ("Phrase was very short. Try sustaining longer next time.", "info")
            elif cat == "short":
                return (f"OK, {last_len_sec:.1f}s phrase. Try for 4-6s.", "info")
            elif cat == "comfortable":
                return (f"Nice phrase! {last_len_sec:.1f}s.", "encourage")
            elif cat == "long":
                return (f"Strong phrase: {last_len_sec:.1f}s.", "encourage")
            elif cat == "very_long":
                return (f"Excellent control: {last_len_sec:.1f}s phrase!", "encourage")
            else:  # running_out
                return (f"Long phrase ({last_len_sec:.1f}s). Make sure you're not straining.", "warn")

        if in_phrase and current_len_sec > 12.0:
            return ("Long phrase in progress. Consider breathing soon.", "warn")
        if in_phrase and current_len_sec > 8.0:
            return (f"Sustaining for {current_len_sec:.1f}s.", "info")
        # Default — quiet
        return ("", "info")

    def _categorize(self, dur_sec: float) -> str:
        for label, (lo, hi) in PHRASE_RANGES.items():
            if lo <= dur_sec < hi:
                return label
        return "very_long"

    def session_summary(self) -> dict:
        """Aggregate stats over all phrases in the session so far."""
        phrases = self.s.phrases
        if not phrases:
            return {
                "n_phrases": 0,
                "total_singing_sec": 0.0,
                "avg_phrase_sec": 0.0,
                "median_phrase_sec": 0.0,
                "max_phrase_sec": 0.0,
                "n_breaths": 0,
            }
        durs = sorted(p.duration_sec for p in phrases)
        n = len(durs)
        median = durs[n // 2] if n % 2 == 1 else 0.5 * (durs[n // 2 - 1] + durs[n // 2])
        return {
            "n_phrases": n,
            "total_singing_sec": sum(durs),
            "avg_phrase_sec": sum(durs) / n,
            "median_phrase_sec": median,
            "max_phrase_sec": max(durs),
            "n_breaths": n,  # one breath per phrase boundary in this simple model
        }


if __name__ == "__main__":
    # Quick smoke test
    import numpy as np
    rng = np.random.default_rng(42)

    # Simulate 60 seconds of frames (6000 frames at 10ms hop):
    # voiced for 5s, breath for 0.3s, voiced for 7s, breath for 0.4s, ... etc.
    n_frames = 6000
    breath_probs = np.zeros(n_frames)
    voiced_probs = np.zeros(n_frames)

    cursor = 0
    pattern = [(5.0, "sing"), (0.3, "breath"),
               (7.0, "sing"), (0.4, "breath"),
               (3.0, "sing"), (0.3, "breath"),
               (10.0, "sing"), (0.5, "breath"),
               (2.0, "sing")]
    for dur_sec, kind in pattern:
        end = min(n_frames, cursor + int(dur_sec / HOP_SEC))
        if kind == "sing":
            voiced_probs[cursor:end] = 0.9
            breath_probs[cursor:end] = 0.05
        else:
            voiced_probs[cursor:end] = 0.1
            breath_probs[cursor:end] = 0.85
        cursor = end

    tracker = PhraseTracker()
    for t in range(n_frames):
        out = tracker.step(breath_probs[t], voiced_probs[t])
        if out.phrase_just_ended:
            print(f"  t={tracker.s.current_time_sec:5.2f}s  "
                  f"phrase ended ({out.last_phrase_length_sec:.2f}s)  → {out.coaching_message}")

    print("\nSession summary:")
    summary = tracker.session_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
