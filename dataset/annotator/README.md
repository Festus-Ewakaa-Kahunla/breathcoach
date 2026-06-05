# labeler/

In-browser breath labeler for the VocalSet-Breath dataset.

- **Pre-fills** every clip with v11's draft breath markers (you confirm/delete/nudge)
- **Keyboard-driven** — full shortcut sheet visible at the bottom of the UI
- **All 5 categories** color-coded (breath, silent_breath, uncertain, hard_negative, exhale)
- **Auto-saves** after every edit (400 ms debounce) directly to `dataset/labels/vocalset/<id>.breath.json`
- **Tracks review time** per clip and **labeler identity** (set once in the sidebar)
- **Spectrogram + waveform** stacked and time-aligned; click anywhere to seek

## Launch

From the repo root:
```bash
.venv/bin/python dataset/tools/labeler/server.py
```

Defaults (override with flags):

| Flag | Default | Notes |
|---|---|---|
| `--port` | 8431 | http://localhost:8431/ |
| `--audio-dir` | `data/_sung_excerpts/` | the 118 sung VocalSet excerpts already staged |
| `--labels-dir` | `dataset/labels/vocalset/` | where `.breath.json` files land |
| `--spec-cache` | `dataset/tools/labeler/.spec_cache` | pre-rendered spectrograms (gitignored) |
| `--nanopitch` | `models/nanopitch/best.pth` | backbone (your finetuned exp12) |
| `--breath-head` | `runs/v11-finetune-gtsinger/best.pth` | model used for draft markers |
| `--prefill-prominence` | 0.22 | how aggressive the v11 draft is (lower = more drafts) |

To label all of VocalSet (not just the staged excerpts):
```bash
.venv/bin/python dataset/tools/labeler/server.py --audio-dir data/vocalset/FULL
```
(walks the full tree; the server lists clips flat. Subset filtering via the sidebar search bar.)

## Workflow

1. Open http://localhost:8431/
2. Enter your name in the **Labeler** field once (saved to localStorage).
3. Click a clip in the sidebar. Draft markers from v11 appear in dashed outline.
4. Press **Space** to play. Use **←/→** (1 s) or **Shift+←/→** (100 ms) to scrub.
5. For each draft marker:
   - **A** to accept (turns solid) — it's a real breath.
   - **D** to delete — it's not.
   - **H** to mark as hard_negative — model wrongly fired here.
   - **U** to mark as uncertain — you're not sure.
6. Press **B** at the playhead to add a breath the model missed.
7. **Tab** to move to the next clip.

Read `dataset/docs/labeling_rubric.md` first — it's the canonical guide for what counts as a breath and what doesn't.

## Output format

Each save writes a `.breath.json` file matching the schema in `dataset/README.md`. Sources track who created each event:

| `source` | Meaning |
|---|---|
| `v11_prefill` | Draft from the model, not yet reviewed |
| `human_accepted` | You confirmed the draft |
| `human_added` | You added it from scratch |
| `human_nudged` | You adjusted the draft's boundaries |
| `human_recategorized` | You changed its category |

A clip is shown as `labeled` in the sidebar once *any* marker has a non-`v11_prefill` source. `draft` means all markers are still raw model output.

## Status / progress

The header shows `N / M labeled · X.X min total` at all times. Save status (`saving…` / `✓ saved` / `⚠ save failed`) is shown next to it.
