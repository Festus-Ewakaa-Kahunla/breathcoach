# Hand-Labeling Guidelines for Singing Breath Events

Project 2 — singing breath/phrase coach. This doc tells me (and any future re-labeler) exactly how to mark up audio files in Sonic Visualizer and export to our JSON format.

## What we are labeling

For each WAV file we mark every **audible inhalation** (a "breath event") that occurs while the singer is producing the singing performance. We do NOT label:
- Silences without an audible inhale (those are pauses, not breaths)
- The singer's resting breathing before the performance starts
- Coughs, sniffs, lip noises, throat clears (mark as a NOTE for later cleanup, but do not flag as breath)

A breath event has two timestamps:
- `start_sec` — the first frame where audible inhalation noise is detectable
- `end_sec` — the last frame where audible inhalation noise is detectable (i.e., the singer has finished inhaling and is about to phonate again)

Typical breath events are 100–500 ms long.

## What counts as "audible inhalation"

Listen for:
- The hiss/rush of air entering through the mouth (sounds like /h/)
- The slight nasal hiss if breath is partly through the nose
- A brief energy spike in the spectrogram at 1.5–3 kHz (breath has a characteristic mid-band noise signature)

If you can't HEAR the breath but you can SEE the spectrogram signature, it counts. If you can hear it but not see it, also counts. Either signal is sufficient.

## Tool: Sonic Visualizer

1. Open the WAV in Sonic Visualizer (`brew install --cask sonic-visualiser` if needed).
2. Add a **Time Instants Layer** for breath events (single time markers — use the START of each breath).
3. (Optional) Add a **Time Values Layer** for breath duration if you want to mark end-times too. Simpler version: just mark starts and let the script estimate durations from energy envelope.
4. Listen at 0.5x speed when uncertain.
5. Always look at the spectrogram (Layer → Add Spectrogram, log-scale frequency, 1024 window).
6. Save as `.svl` from Sonic Visualizer's File menu (it's an XML format).

## Export format (our canonical JSON)

For each labeled audio file, save a sibling `.breath.json` next to the WAV:

```json
{
  "audio_file": "f1_arpeggios_breathy_a.wav",
  "sample_rate": 16000,
  "duration_sec": 12.34,
  "labeler": "Festus",
  "label_date": "2026-05-08",
  "breath_events": [
    { "start_sec": 1.83, "end_sec": 2.06, "confidence": "high" },
    { "start_sec": 4.71, "end_sec": 4.95, "confidence": "high" },
    { "start_sec": 9.12, "end_sec": 9.41, "confidence": "low" }
  ],
  "notes": "Some lip noise around 7.2s — not a breath."
}
```

### Field semantics
- `start_sec`, `end_sec` — float seconds, breath onset/offset
- `confidence` — `"high"` if I'm sure it's a breath, `"low"` if I'm guessing (lets us filter later)
- `notes` — anything weird about the recording or my labels

If I only marked start times (faster workflow), set `end_sec = start_sec + 0.25` as a default and write `notes: "end_sec auto-estimated"`.

## Rate target

- 1× listen-through (~1× audio time)
- ~30s of focused labeling per 1 min of audio (= 0.5× labeling time)
- Hard cap: **4 hours total labeling**, regardless of how much audio we get through

So 4 hours of focused labeling at 1.5× total time per minute of audio = up to ~160 minutes (~2.5 hours) of labeled audio if I'm fast. Realistic target is 20–30 minutes labeled.

## Inter-session check

After I label the first 10 min, I'll:
1. Wait at least one day
2. Re-label the same first 5 min from scratch (without looking at the original labels)
3. Compare: how many breath events match within 100ms tolerance?

This gives us inter-session reliability (a proxy for inter-annotator agreement, which we don't have because solo project). Report this number in the paper.

## What "good" looks like

A well-labeled 30-second clip should have:
- Every breath event marked
- No false positives (I'm not flagging silences as breaths)
- Confidence levels honestly assigned (most should be "high"; "low" is for the genuinely ambiguous ones)
- Notes for any oddities
