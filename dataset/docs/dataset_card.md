---
license: cc-by-4.0
task_categories:
  - audio-classification
  - voice-activity-detection
language:
  - en
tags:
  - singing
  - breath-detection
  - music-information-retrieval
  - singing-voice
  - audio-annotation
pretty_name: VocalSet-Breath
size_categories:
  - n<10K
---

# Dataset Card — VocalSet-Breath

## Dataset summary

VocalSet-Breath is a **hand-labeled annotation layer** on top of [VocalSet (Wilkins et al., 2018)](https://zenodo.org/records/1193957) that adds time-aligned audible-breath-event labels to every clip. It is the first publicly redistributable singing-voice corpus with frame-level breath-event ground truth, intended to enable training and evaluation of audible-breath-detection models for singing.

## Languages

English. (VocalSet base corpus is English vowel exercises + English-text excerpts.)

## Supported tasks

- **Breath-event detection** in singing voice (frame-level or event-based)
- **Phrase segmentation** in singing voice
- **Hard-negative learning** — explicit fricative / lip-noise / `/h/` negatives for precision-aware training
- **Cross-corpus evaluation** when combined with speech-domain breath corpora (Respiro-en, etc.) or with GTSinger

## Source data

- **Audio**: VocalSet (Wilkins et al., ISMIR 2018). 10.1 hours, 20 singers, 17 vocal techniques. Not redistributed here — fetch from Zenodo (DOI: 10.5281/zenodo.1193957).
- **Annotations** (this work): hand-labeled by Festus Ewakaa Kahunla following the [labeling rubric](labeling_rubric.md). Pre-filled from a neural breath detector ("v11") and reviewed clip-by-clip.

## Annotations

Per `.breath.json` file:

| Field | Type | Description |
|---|---|---|
| `audio_file` | string | VocalSet wav filename |
| `sample_rate` | int | 16000 (resampled) |
| `duration_sec` | float | clip duration |
| `labeler` | string | who labeled |
| `label_date` | string | YYYY-MM-DD |
| `tool_version` | string | labeling tool version |
| `review_time_sec` | int | seconds spent labeling this clip |
| `pre_fill_source` | string | model checkpoint that produced draft markers |
| `breath_events` | list | confirmed audible inhalations |
| `silent_breaths` | list | visible on spectrogram but inaudible |
| `uncertain` | list | low-confidence events |
| `hard_negatives` | list | explicit negatives (fricative, lip noise, etc.) |
| `exhales` | list | audible outward breaths |
| `notes` | string | free-form clip-level notes |

Each event has `start_sec`, `end_sec`, optional `confidence` (`high`/`medium`/`low`), optional `notes`.

## Splits

By-singer splits (no leakage). Singer IDs in `splits/{train,val,test}.txt`. Generated reproducibly via `tools/make_splits.py` with a locked seed.

## Annotation process

1. Audio loaded into in-browser labeling tool with **pre-filled markers** from a neural breath detector.
2. Labeler reviews each marker (accept / delete / nudge) using spectrogram + waveform + audio playback.
3. Labeler adds missed breaths, marks hard negatives, tags uncertain cases.
4. Quality discipline: anti-drift practices (calibration, edge-case log) and inter-annotator agreement on a ≥15% subset (Cohen's κ + event-F1). See [labeling rubric](labeling_rubric.md) §5.

## Considerations

- **Bias / scope**: VocalSet is English vowel exercises + short song excerpts performed by 20 professional singers in studio conditions. It is **not** representative of amateur singing, popular music with accompaniment, non-English languages, or in-the-wild recording conditions. Models trained on it will need cross-corpus eval (e.g., on GTSinger or new wild recordings) to validate generalization.
- **Annotator effect**: single primary labeler; the IAA subset partially mitigates but does not eliminate annotator bias. Future versions will add additional labelers.
- **License of base data**: VocalSet is CC BY 4.0 and Wilkins et al. 2018 must be cited.

## Citation

See [`CITATION.cff`](../CITATION.cff).

## Maintainer

Festus Ewakaa Kahunla, Drexel University.
