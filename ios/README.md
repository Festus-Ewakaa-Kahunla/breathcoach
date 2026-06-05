# BreathCoach — iOS

A native SwiftUI app that runs the BreathCoach model **fully on-device** via CoreML —
no server, works offline. Record (or load a clip) and it marks where you breathed, your
phrase lengths, and a per-phrase coaching message, over the analyzed waveform and spectrogram.

Built with SwiftUI + iOS 26 (Liquid Glass), Swift 6, iPhone-only.

> **Not real-time:** you record (or load a clip), the model analyzes it, then it plays
> back with the breath + phrase overlay. Streaming is future work.

## Requirements

- macOS with **Xcode 26** (Swift 6, iOS 26 SDK)
- **XcodeGen** — `brew install xcodegen` (the Xcode project is generated from `project.yml`)

## Run

```bash
cd ios
xcodegen generate            # creates BreathCoach.xcodeproj from project.yml
open BreathCoach.xcodeproj
```

Then in Xcode:

1. **Simulator** — pick an iPhone (iOS 26) and press **Run (⌘R)**. No signing needed.
2. **Physical iPhone** — open **Signing & Capabilities** and set **your own** Development
   Team (the `DEVELOPMENT_TEAM` in `project.yml` is the author's), then Run. On first launch,
   trust the profile on the phone under *Settings → General → VPN & Device Management*.

The CoreML model (`BreathCoach/Resources/Model/BreathCoach.mlpackage`) is bundled, so all
inference runs on the phone — no network, no `serve.py`.

## How it works

- `Services/LogMel.swift` — log-mel features via vDSP (mirrors the Python preprocessing).
- `Services/OnDeviceBreath.swift` — runs the CoreML model, then peak-picks → breath events → phrases.
- `Models/CoachController.swift` — the record / upload / demo-clip flows and playback.

Details in [`docs/coreml_integration.md`](docs/coreml_integration.md).

## Modes

- **Coach** — record or load a clip → breath markers, phrase length, a coaching banner, and
  an expandable spectrogram + probability view. **Working.**
- **Mimic** — sing-along comparison against a reference track. **Planned — currently stubbed.**

## Screenshots

See [`design/shots/`](design/shots), or the main [README](../README.md#screenshots).
