# dataset/

The **annotation tool** used to build **VocalSet-Breath** — a hand-labeled
breath-event layer on top of [VocalSet](https://zenodo.org/records/1193957).

The labels themselves are **not** in this repo. They live on Hugging Face,
because the dataset is still growing:

**→ https://huggingface.co/datasets/Ewakaa/Vocalset-Breath**

## What's here

- `annotator/` — the labeling tool. `index.html` (waveform + spectrogram review UI)
  and `server.py` (serves clips, writes `.breath.json`). It pre-fills breath
  guesses from the model; the labeler confirms, fixes, or adds.
- `docs/dataset_card.md` — the dataset card (also the README on Hugging Face).
- `docs/labeling_rubric.md` — what counts as a breath, confidence levels, hard negatives.

## Run the annotator

```bash
cd annotator
python server.py            # then open the printed localhost URL
```

Point it at your local VocalSet audio — see `annotator/README.md`.

## Label schema

Each clip → one `.breath.json` with `breath_events`, `silent_breaths`,
`hard_negatives`, `exhales`, and `uncertain` lists — each entry
`{start_sec, end_sec, confidence, source}` — plus labeler and provenance metadata.
Full description in the dataset card.
