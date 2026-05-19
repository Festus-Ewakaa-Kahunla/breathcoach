# scripts/

One-off CLI utilities. Library code lives in `src/nanobreath/`; this directory is for thin wrappers and pipeline glue.

| Script | Purpose |
|---|---|
| `smoke_test_pipeline.py` | End-to-end sanity check: load a WAV, compute mel, run BreathHead forward, run the Ruinskiy baseline, run the phrase tracker. No backbone or labels required. Use this first after a fresh checkout. |
| `generate_pseudo_labels.py` | Run the Ruinskiy & Lavner baseline over a directory of WAVs and emit `*.breath.json` sidecars with `confidence: "low"`. Use this for the Respiro-en-style self-training warmup before hand-labels are ready. |

Run from the repo root after `pip install -e .`:

```bash
python scripts/smoke_test_pipeline.py
python scripts/generate_pseudo_labels.py \
    --audio-root data/vocalset/FULL \
    --out-dir   data/labels/pseudo \
    --limit     20
```
