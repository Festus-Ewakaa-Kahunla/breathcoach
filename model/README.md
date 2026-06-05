# model/

The model behind BreathCoach: a **frozen NanoPitch pitch backbone** with a small
**trained BreathHead** on top.

## Architecture

```
audio ─► log-mel (40-band, 10 ms hop) ─► NanoPitch [FROZEN, ~333K params]
      ─► 384 features/frame ─► BreathHead [TRAINED, ~15K params] ─► p(breath)/frame
```

Only the BreathHead is trained. The backbone — already trained for pitch and
voicing — is frozen and used as a feature extractor. Because the hard part
(understanding the singing voice) is reused, breath detection learns from a small
amount of labeled data, and the whole model is small enough to run on a phone.

## Layout

- `src/nanobreath/model/` — `joint.py` (frozen backbone + head), `breath_head.py` (the causal head)
- `src/nanobreath/train.py`, `eval_events.py` — training + event-based F1 (DCASE, ±200/±100 ms)
- `src/nanobreath/baseline/ruinskiy_lavner.py` — the 2007 DSP baseline
- `src/nanobreath/deployment/` — `export_breath_head.py` (CoreML / WASM), `precompute_predictions.py`
- `coreml/` — exported `BreathCoach.mlpackage` + `preprocessing.json` + `parity_reference.json`

## Install

```bash
pip install -e .
```

## Weights

Included, so the project runs out of the box:

```
model/weights/nanopitch/best.pth     # frozen NanoPitch backbone
model/weights/nanopitch/model.py     # the NanoPitch class (architecture)
model/weights/breathcoach_v13.pth    # the trained joint model (production)
model/coreml/BreathCoach.mlpackage   # CoreML export (used by the iOS app)
```

Override paths with `$NANOPITCH_CHECKPOINT` / `$NANOPITCH_SRC_DIR` if needed.

## Results — held-out singer, event-F1 @ ±200 ms

| Method | F1 | Recall |
|---|---|---|
| Ruinskiy & Lavner 2007 (DSP, no learning) | 0.01 | 0.4% |
| BreathCoach | 0.67 | 97% |
