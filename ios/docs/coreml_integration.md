# On-device CoreML integration — Coach

**Goal:** run breath detection entirely on the phone. No server, no Wi-Fi, no "server unreachable" in front of anyone. The model is proven-converted and validated against the Python pipeline.

This replaces the `BreathAPI.process()` network call in Coach with a local CoreML inference. **Keep `BreathAPI` in the codebase as a fallback** (a build flag or settings toggle) — don't delete it.

---

## What's delivered

All in the research repo at `../breathcoach/models/coreml/`:

| File | What it is | Where it goes |
|---|---|---|
| `BreathCoach.mlpackage` | The model (468 KB). Input `mel`, outputs `breath` + `voiced`. | Add to the Xcode target (bundle it) |
| `preprocessing.json` | Exact log-mel params + the 40×257 mel filterbank + Hann window | Add to the app bundle; load at launch |
| `parity_reference.json` | `sample.wav`'s expected mel + model outputs (first 200 frames) | Dev-only — for the parity test, don't ship |

Copy the first two into the Xcode project (e.g. `BreathCoach/Resources/`). Xcode auto-compiles `.mlpackage` into a `BreathCoach` Swift class.

---

## The model (CoreML I/O)

```
input   mel     : MLMultiArray  shape (1, T, 40)   float32   — log-mel, T = number of 10ms frames
outputs breath  : MLMultiArray  shape (1, T, 1)    float32   — p(breath) per frame, 0..1
        voiced  : MLMultiArray  shape (1, T, 1)    float32   — p(voiced) per frame (for the silence gate)
```

T is flexible (10–6000 frames → 0.1 s to 60 s). Compute units: `.all` (CPU/GPU/ANE).

---

## New Coach data flow (replaces the /process POST)

```
record (AVAudioEngine)  →  Float32 PCM
   │  (resample to 16 kHz mono if needed — AVAudioConverter)
   ▼
1. LOG-MEL  → MLMultiArray (1, T, 40)        ← the parity-critical step, see below
   ▼
2. CoreML BreathCoach.prediction(mel:)  → breath[T], voiced[T]
   ▼
3. PEAK-PICK breath[]  → breath events (start_sec, end_sec)
   ▼
4. breaths → phrases → existing CoachingState / banner / timeline (unchanged)
```

Everything downstream of step 2 is the logic you already have in `CoachController` — it just gets its `breath`/`voiced` arrays from CoreML instead of `ProcessResponse`.

---

## Step 1 — Log-mel in Swift (do this exactly; it's the only real risk)

Load `preprocessing.json` once. It contains `mel_filterbank` (40×257), `hann_window` (400), and the params. Per frame:

```
for each frame t (t = 0, 160, 320, … while start+400 <= n_samples):
    frame   = audio[start ..< start+400]              // 400 samples (25 ms)
    frame   = frame .* hann_window                     // elementwise, bundled window
    spec    = rfft(frame, n=512)                        // 257 complex bins — use vDSP
    power   = spec.real^2 + spec.imag^2                 // |spec|^2, length 257
    mel     = mel_filterbank (40×257) * power           // matrix·vector → 40 values (vDSP_mmul)
    logmel  = log(mel + 1e-10)                          // 40 values → row t of the (T,40) input
```

- FFT: `vDSP`/`Accelerate` `vDSP_DFT_Execute` or `vDSP.FFT`, size 512, real input (zero-pad the 400-sample windowed frame to 512).
- **Do not regenerate the mel filterbank in Swift** — use the matrix from `preprocessing.json` verbatim. That's what guarantees parity.
- `np.hanning(400)` is a 400-pt symmetric Hann — use the bundled `hann_window` array, don't recompute.
- Frame count: `T = (n_samples - 400) / 160 + 1`.

## Step 3 — Peak-pick (port of `peak_events`, defaults)

```
smooth     = movingAverage(breath, window = round(0.080 / 0.010) = 8 frames)   // ~80 ms
baseline   = median(smooth)
peaks      = local maxima of smooth with:
               prominence >= 0.12
               min distance >= round(1.0 / 0.010) = 100 frames apart   // 1 s
for each peak p:
   cutoff  = baseline + (smooth[p] - baseline) * 0.6
   walk left/right from p until smooth < cutoff  → event [start, end]
event.start_sec = startFrame * 0.010 ;  end_sec = endFrame * 0.010
```

(There's no `find_peaks` in Swift — implement local-max + prominence + min-distance directly. The Python fallback scan in `precompute_predictions.py` lines 154–157 is a good reference for the simple version.)

## Step 4 — phrases + coaching

Unchanged from what Coach already does: a phrase = the span between consecutive breaths; `CoachingState.compute` and the banner/timeline consume `breath`/`voiced` exactly as before. The **voiced gate** (only coach when `voiced >= 0.4`) you already have — feed it the `voiced` output.

---

## Parity test (do this before trusting on-device)

`parity_reference.json` has `sample.wav`'s expected `mel_first_200` (200×40) and `breath_first_200`.

1. Bundle `sample.wav` (from `../breathcoach/src/nanobreath/deployment/web/sample.wav`), load it, compute your Swift log-mel.
2. Assert your `mel[0..<200]` matches `mel_first_200` within ~1e-3. **If this fails, fix preprocessing before anything else** — a mel mismatch is the whole ballgame.
3. Run CoreML on your mel; assert `breath[0..<200]` matches `breath_first_200` within ~0.02 (FP16 tolerance).

If both pass, on-device equals the validated server. Ship it.

---

## Notes

- Model = **v13** (the production model: F1 0.669 on the held-out singer). If we retrain (v15 from 250 labels), I'll hand you a new `.mlpackage` — same I/O, drop-in replace.
- The 0.003 FP16 difference vs PyTorch is below the peak-pick threshold — it does not change detected breaths.
- Recording length is capped by the model's `upper_bound` (6000 frames = 60 s). Coach records ≤ 30 s, so fine.
- Keep `BreathAPI` behind a toggle (`USE_ON_DEVICE = true`) so you can A/B and fall back if needed.
