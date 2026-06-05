# web/

Browser demo. Record (or load a clip) and see detected breaths on the
spectrogram, with phrase-length feedback.

## Run

```bash
# 1. install the model package
cd ../model && pip install -e .
# 2. start the server (serves this folder + a /process endpoint)
cd ../web && python serve.py            # http://localhost:8421
```

## Weights

Included and wired by default — `serve.py` loads:

- backbone → `../model/weights/nanopitch/best.pth`
- head     → `../model/weights/breathcoach_v13.pth`

so `python serve.py` works with no flags. Override with `$NANOPITCH_CHECKPOINT`
or `--breath-head` for a different checkpoint.

## Demo clips

Three held-out demo clips live in `clips/` — GTSinger Tenor-1 renditions, included
under CC BY-NC-SA (attribute GTSinger). The model never trained on them. Use the
**Demos** menu to load one, or the **Record** button for your own voice.
