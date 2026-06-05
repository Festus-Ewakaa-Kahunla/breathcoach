<div align="center">

<img src="docs/assets/hero.svg" alt="BreathCoach - breath & phrase coaching for singers" width="100%">

<p>
  <a href="#run-it"><img alt="Run the demo" src="https://img.shields.io/badge/Run%20the%20demo-2ee6c8?style=for-the-badge&labelColor=0b1020"></a>
  <a href="#how-it-works"><img alt="How it works" src="https://img.shields.io/badge/How%20it%20works-6c7bff?style=for-the-badge&labelColor=0b1020"></a>
  <a href="#screenshots"><img alt="Screenshots" src="https://img.shields.io/badge/Screenshots-ff5ea0?style=for-the-badge&labelColor=0b1020"></a>
</p>

<p>
  <img alt="params" src="https://img.shields.io/badge/params-15%2C705-2ee6c8?style=flat-square&labelColor=0b1020">
  <img alt="latency" src="https://img.shields.io/badge/latency-~0.05%20ms%2Fframe-6c7bff?style=flat-square&labelColor=0b1020">
  <img alt="val PR-AUC" src="https://img.shields.io/badge/val%20PR--AUC-0.65-ff5ea0?style=flat-square&labelColor=0b1020">
  <img alt="ECE" src="https://img.shields.io/badge/ECE-0.022-2ee6c8?style=flat-square&labelColor=0b1020">
  <img alt="python" src="https://img.shields.io/badge/python-3.11-6c7bff?style=flat-square&labelColor=0b1020&logo=python&logoColor=white">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-frozen%20backbone-ff5ea0?style=flat-square&labelColor=0b1020&logo=pytorch&logoColor=white">
  <img alt="license" src="https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-8b93a7?style=flat-square&labelColor=0b1020">
</p>

</div>

On-device breath detection for singing. A small causal network (~15K trainable
parameters) detects audible breaths frame-by-frame on top of a **frozen**
pitch-tracking backbone (NanoPitch), and turns the result into phrase-length
feedback for singers.

## The parts

| Folder | What it is |
|---|---|
| `model/` | The model - frozen NanoPitch backbone + trained BreathHead. Training, evaluation, CoreML export. |
| `dataset/` | The annotation tool used to build **VocalSet-Breath**. The labels live on Hugging Face. |
| `web/` | Browser demo - record or load a clip, see breaths on the spectrogram. |
| `ios/` | Native iOS app (SwiftUI). Runs the model fully on-device via CoreML. |

## Datasets

- **VocalSet-Breath** (ours) - https://huggingface.co/datasets/Ewakaa/Vocalset-Breath
- **GTSinger** (training) - https://github.com/AaronZ345/GTSinger

`dataset/annotator/` is the tool that produced VocalSet-Breath.

## How it works

```
audio ─► log-mel ─► NanoPitch [FROZEN, ~333K] ─► 384 features/frame
      ─► BreathHead [TRAINED, ~15K] ─► p(breath)/frame ─► breath + phrase feedback
```

Only the 15K-parameter head is trained. The backbone - already trained for pitch
and voicing - is frozen and used as a feature extractor. Reusing it means breath
detection learns from a small amount of labeled data, and the whole model is
small enough to run on a phone.

## Screenshots

**iOS - fully on-device**

<p>
  <img src="ios/design/shots/10-home.png"          width="30%" alt="Home">
  <img src="ios/design/shots/12-coach-playing.png" width="30%" alt="Coach - breath timeline over the waveform">
  <img src="ios/design/shots/13-coach-ended.png"   width="30%" alt="Session summary">
</p>

**Web demo**

<p>
  <img src="docs/screenshots/web/web-coach.png" width="80%" alt="Browser demo - detected breaths marked on the waveform">
</p>
<p>
  <img src="docs/screenshots/web/web-analysis.png" width="80%" alt="Analysis - log-mel spectrogram, breath probability vs. the 2007 DSP baseline, and model meta">
</p>

## Run it

**Model + web demo**
```bash
cd model && pip install -e .     # installs the `nanobreath` package
cd ../web && python serve.py     # http://localhost:8421
```

**iOS**
```bash
cd ios && xcodegen generate      # then open BreathCoach.xcodeproj in Xcode and Run
```

## License & weights

- **Code & model** - CC BY-NC-ND 4.0 (see `LICENSE`).
- **Model weights** - backbone, trained head, and CoreML package are included
  (`model/weights/`, `model/coreml/`) for research and demo use.
- **VocalSet-Breath labels** - CC BY 4.0 (VocalSet is CC BY 4.0). Hosted on Hugging Face.
- **Training data** - the model trains on [GTSinger](https://github.com/AaronZ345/GTSinger)
  (Zhang et al., NeurIPS 2024; CC BY-NC-SA). The demo clips are held-out GTSinger
  renditions under the same license; the full corpus is not redistributed here.

## Credits

Festus Ewakaa Kahunla - Drexel, Musical AI, 2026.
Built on NanoPitch by Smule (https://github.com/smulelabs/NanoPitch).
Training data: [GTSinger](https://github.com/AaronZ345/GTSinger) (Zhang et al., NeurIPS 2024).
Baselines: Ruinskiy & Lavner (2007); Respiro.
