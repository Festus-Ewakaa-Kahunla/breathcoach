# VocalSet-Breath — Labeling Rubric

> **The canonical guide for anyone labeling breath events in singing-voice audio for this dataset.** Follow it strictly; deviations create silent drift that breaks training. If you encounter an edge case not covered here, **decide once, write the rule in the edge-case log at the bottom of this file, and follow it from then on.**

---

## 1. What a breath is — the three fingerprints

A real **audible inhalation** has three signatures you check **simultaneously**. A label is "high confidence" only when 2 or 3 of them agree.

### A. Spectrogram (visual)
- Broadband noise smudge concentrated in **1.5–3 kHz** (peak ~1.6 kHz male / ~1.7 kHz female, per Nakano et al. 2008).
- **No harmonic stripes** — those are voiced phonation.
- **No formant pattern** — those are vowels.
- Energy roughly uniform across the breath band.

### B. Waveform (energy)
- Lower than singing, but **not silent**.
- Typically **10–20 dB above the silence floor**.
- Often has a soft attack and a tapered end (no sharp click).

### C. Ear (audio)
- A `/h/`-like rush of air.
- Duration typically **100–500 ms** (full range observed in literature: 50–1225 ms).
- Located **between phrases**, almost always preceding the next sung note.
- Sometimes nasal-tinted (breath through nose).

### Decision rule
| Signatures present | Action |
|---|---|
| 3 of 3 | `breath`, `confidence: high` |
| 2 of 3 | `breath`, `confidence: medium` |
| 1 of 3 | `uncertain` (or not a breath at all — see §3 confusables) |
| 0 of 3 | Not a breath. Skip or mark as `hard_negative` if model is firing here |

---

## 2. The 5 categories — concrete definitions

### 2.1 `breath`
A confirmed audible inhalation. Mark:
- **`start_sec`**: first frame where you can detect inhale hiss (look at spectrogram + listen).
- **`end_sec`**: last frame of inhale hiss before phonation resumes.
- **`confidence`**: `"high"` | `"medium"` (low confidence → use `uncertain` instead).
- **`notes`**: optional, e.g., `"deep breath, end of verse"` or `"sneaky catch breath"`.

### 2.2 `silent_breath`
Visible on spectrogram (broadband mid-frequency smudge) but **inaudible** to the ear. Common with professional singers. Mark with the same `start_sec`/`end_sec`, but **never used as a positive in detection eval** — labeled separately for research interest.

### 2.3 `uncertain`
You're not sure. One signature says breath, others don't. Mark anyway with a `notes` field explaining why:
```json
{"start_sec": 12.41, "end_sec": 12.55, "notes": "soft hiss but could be /h/ onset"}
```
We filter these out at training time (or weight them lower). Don't agonize — mark uncertain and move on.

### 2.4 `hard_negative` ⭐
Sounds or looks like a breath but is **NOT**. The model would fire here and be **wrong**. **This is the most valuable class for precision training** — explicit negatives anchor the model. Examples:
- Fricatives: `/s/`, `/f/`, `/ʃ/`, `/θ/`
- Lip smacks, mouth opening clicks
- Throat clears, coughs
- Word-initial `/h/` ("hello", "high")
- Sustained `/h/` in lyric
- Audible mic noise, room noise

Mark with `notes` explaining what it is.

### 2.5 `exhale`
Audible outward breath. Distinguishable from inhalation by:
- Often occurs **after** a phrase ends (whereas inhale precedes the next phrase).
- May have residual voicing decay (slight harmonic content tapering).
- Spectral peak often lower in frequency than inhalation.

Less common in formal singing; show up in expressive/breathy styles.

---

## 3. Confusables — what NOT to mark as `breath`

| What it looks like | What it actually is | How to tell |
|---|---|---|
| Mid-frequency noise | **Fricative** `/s/`, `/f/`, `/ʃ/` | Embedded inside a word, has voiced neighbors |
| `/h/`-like hiss | **Word-initial** `/h/` ("hello", "heart") | Followed immediately by a vowel with formants |
| Broadband noise | **Sustained** `/h/` or breathy vowel | Weak harmonic structure still visible |
| Short noise burst | **Mic click**, mouth smack, plosive release | Very short (<50 ms), percussive shape |
| Long flat noise | **Silent pause + mic noise floor** | Energy only in 200–1000 Hz, no mid-band smudge |
| Soft inhale-like sound | **Audible exhale** | Comes *after* phrase, lower freq peak |

**When in doubt**: spectrogram > ear. The spectrogram is more reliable than your ears for the 1.5–3 kHz smudge.

---

## 4. Workflow — how to actually do it efficiently

For each clip in the labeling tool:

1. **Listen once at 1× speed.** Get the song's shape in your head — where the phrases are, where the singer breathes intentionally.
2. **Scan the spectrogram** for the broadband smudges in 1.5–3 kHz. Cross-check with the waveform.
3. **Walk through the v11 pre-fill markers in order**:
   - **Accept** (`A`) if 2+ fingerprints agree → confirmed.
   - **Delete** (`D`) if it's a confusable (see §3).
   - **Nudge** (`[` / `]`) if onset/offset are clearly off.
4. **Scan unmarked regions** for misses — especially right *before* each phrase starts. Press `B` to add.
5. **Tag obvious model-failure spots** as `hard_negative` (`H`). This is where the model fires but shouldn't. **Don't skip these — they're the most valuable signal.**
6. **Ambiguous moments**: drop playback speed to **0.5×**, loop the region (`L`), look at the spectrogram carefully. Decide → `breath` / `uncertain` / `hard_negative` / not-a-breath.
7. **Save** (`S`). Move to next clip (`Tab`).

### Target rate
- ~**1.5× audio duration** per labeling pass (10-min clip → ~15 min of labor).
- Faster → you're rushing, will miss subtleties.
- Much slower → tool is fighting you (report it).

### Sit length
- **No more than ~45 minutes continuous.** Take a real break.
- Standards drift silently with fatigue.

---

## 5. Quality discipline — what makes this research-grade

### 5.1 Anti-drift practices

**Weekly calibration.** Pick 3 random clips from your earlier sessions. Re-label them *without seeing the old labels*. Compute event-F1 between the two passes (the tool has a button). This is your **inter-session reliability** — defensible in the paper as the ceiling on annotation precision.

**Edge-case log.** When you hit a new ambiguity, decide once and write the rule at the bottom of this file (§7). Future-you and any second labeler follow it.

**Gold clips.** First week: label 5 reference clips together (with discussion). These become your standard. Re-label them every couple weeks → if your F1 against them drifts, you've drifted.

### 5.2 Inter-annotator agreement

For a real dataset release, **at least one other person must label ≥15% of clips independently**. The tool exports both versions; `tools/compute_iaa.py` computes:
- **Cohen's κ** on framed labels (standard inter-annotator metric)
- **Event-F1** between annotators at ±200 ms tolerance (the SED convention)

A κ above 0.7 or event-F1 above 0.85 is the bar for a publishable annotation campaign. We **report this number in the dataset paper**, no matter what it is — it's the floor reviewers care about.

### 5.3 Common drift modes to watch for

| Drift | Sign |
|---|---|
| **Tightening** | "I only count super-obvious breaths now" — F1 vs gold falls because recall drops |
| **Loosening** | "Anything fuzzy is a breath" — F1 vs gold falls because precision drops |
| **Boundary creep** | Onsets/offsets drift earlier or later — IoU drops while event-F1 holds |
| **Category drift** | You start marking exhales as breaths, or hard negatives as uncertain |

---

## 6. Resources

### Spectrogram reading
- Rob Hagiwara, [*How to Read a Spectrogram*](https://home.cc.umanitoba.ca/~robh/howto.html) — canonical intro
- Hugo Quené, [*Tutorial on Phonetics and Speech Analysis* Ch. 6](https://hugoquene.github.io/TPhSA-EN/ch-spectrograms.html)
- [PhonaLab — wideband vs narrowband spectrograms](https://www.phonalab.com/en/guides/posts/spectrogram-reading)

### Audio-annotation methodology
- [Toloka — Audio data labeling guide](https://toloka.ai/blog/audio-data-labeling-the-complete-guide/)
- [Seshat (arXiv:2003.01472)](https://arxiv.org/pdf/2003.01472) — managing audio annotation campaigns

### Breath acoustics in singing (primary literature)
- Nakano et al. 2008, *Analysis and Automatic Detection of Breath Sounds in Unaccompanied Singing Voice* — the canonical paper. Spectral peak data (1.6–1.7 kHz). [PDF](https://staff.aist.go.jp/m.goto/PAPER/ICMPC2008nakano.pdf)
- Ruinskiy & Lavner 2007, *An Effective Algorithm for Automatic Detection and Exact Demarcation of Breath Sounds in Speech and Song Signals* — DSP-feature breakdown
- Yang et al. 2024, *Frame-Wise Breath Detection with Self-Training* (Respiro-en), Interspeech 2024 — [arXiv](https://arxiv.org/abs/2402.00288), [demo audio](https://ydqmkkx.github.io/breath-detection/) — listen to clean speech breaths next to their spectrograms

### Vocal pedagogy — *understanding* what singers do
Knowing **what kinds of breaths singers take on purpose** helps you not miss the quiet ones:
- "Catch breath" vs "full breath" vs "low breath / appoggio" — YouTube channels *New York Vocal Coaching*, *Eric Arceneaux*, *Veronica Vox* demo each clearly
- Why this matters: a sneaky catch breath is very quiet but still audible — you'd miss it if you only listened for the obvious ones

### Tools
- Praat — free, mature, deep features (TextGrid format we also export). Joseph Casillas' YouTube tutorial series is the quickest intro.
- Sonic Visualiser — easy spectrogram viewing if you want to study examples outside the labeling tool.

---

## 7. Edge-case log

> **Append a new entry every time you hit a case the rubric above doesn't clearly cover. Date it. Include a clip reference and the rule you applied.**

Example entry format:
```
### 2026-05-28 — Soft inhale through closed teeth
- Clip: vocalset/f3_arpeggios_breathy_a.wav, ~3.2s
- Issue: very quiet hiss, looks like spectrogram smudge but barely audible
- Rule: if visible on spectrogram + audible at 0.5× speed, mark as `breath` with confidence `medium`
- Rationale: at 1× the singer hears it, the listener hears it on careful playback
```

(empty for now)

---

## 8. Version history

- **2026-05-26 — v1.0** — Initial rubric. Five categories, anti-drift discipline, IAA protocol.

---

## Quick reference card

| Key | Action | Category |
|---|---|---|
| `A` | accept pre-fill | `breath` |
| `D` | delete | (removes from set) |
| `B` | new marker at playhead | `breath` |
| `H` | mark as hard-negative | `hard_negative` |
| `U` | mark as uncertain | `uncertain` |
| `S` | mark as silent breath | `silent_breath` |
| `E` | mark as exhale | `exhale` |
| `1`/`2`/`3` | confidence high/medium/low | — |
| `[` / `]` | nudge start/end by ±10 ms | — |
| `L` | loop selection | — |
| `Z`/`Shift+Z` | undo / redo | — |
| `Tab` / `Shift+Tab` | next/prev clip | — |
| `Space` | play/pause | — |
| `←`/`→` | scrub 1 s | — |
| `Shift+←`/`→` | scrub 100 ms | — |
