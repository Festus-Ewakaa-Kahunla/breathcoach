# Coach screen — UI spec

> **Layout decision (2026-05): wave-first, no hero ring.** §4.2 below specs a
> circular hero ring as the centerpiece. We prototyped that against a wave-only
> layout (the analyzed waveform as the hero) and chose **wave-only** — it shows
> the whole breathing pattern at once and needs less scrolling. The ring
> (`HeroRingView`) has been removed. Treat §4.2 as historical; the rest of the
> spec (banner copy, timeline, summary, analysis, coaching-state math) is live.

> **Architecture update (2026-06): on-device, no server.** This spec was written
> when inference ran on a FastAPI server (`POST /process`). The app now runs the
> model **fully on-device** via CoreML — see [`coreml_integration.md`](coreml_integration.md).
> References to "the server" / "POST /process" below describe that original
> prototype and are historical; the request → response *shape* still matches what
> `OnDeviceBreath` produces locally.

**Mission**: bring the existing web app's Coach experience to native iOS, translated into iOS 26 Liquid Glass. **Mimic mode is not in this iteration — leave the route stubbed.**

The web app already does everything the Coach view needs to do. This doc tells you exactly *what* to mirror; the source-of-truth for behavior is the working web build at:

```
../breathcoach/src/nanobreath/deployment/web/index.html
```

Open that file alongside this spec. Every component below has a working JS reference there.

---

## 1. What Coach is for

A singer:
1. Taps record, sings 5–30 seconds.
2. The app sends the recording to the FastAPI server (`/process`).
3. The server returns: full per-frame breath probability, detected breath events, derived phrase events, log-mel spectrogram PNG, and metadata.
4. The app plays the recording back, animating a coaching layer over a waveform + breath timeline, with a live phrase-duration ring and a contextual message banner.
5. When playback ends, a session summary appears + an expandable analysis section shows spectrogram, waveform, probability curve, and model meta.

This screen is the headline of the demo. Everything visible to the singer in the web app must be visible here, just dressed in Liquid Glass.

---

## 2. Screen anatomy

```
┌───────────────────────────────────────────────┐
│  ← back        Coach          (status dot)   │  ← nav bar
├───────────────────────────────────────────────┤
│                                               │
│   ▶ Play     ↺ Reset     🎤 Record           │  ← controls row
│                                               │
│   ┌─────┐    ╭─────────╮    ┌─────┐          │
│   │ —   │    │  9.2s   │    │  3  │          │  ← stage
│   │last │    │ singing │    │brth │          │  (side stats + hero ring)
│   └─────┘    ╰─────────╯    └─────┘          │
│                                               │
│   ╭───────────────────────────────────────╮   │
│   │   Comfortable, well-supported         │   │  ← coach banner
│   │   phrasing.                           │   │  (color = quality)
│   ╰───────────────────────────────────────╯   │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │ Your breathing pattern    0.00 / 12.4 │   │
│   │  ═══▂▃▄▅▆▅▄▃═════●═════▃▅▆══════     │   │  ← timeline card
│   │  legend                                │   │
│   └───────────────────────────────────────┘   │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │ Session summary                       │   │
│   │  4   8.2s   12.1s   2.3s              │   │  ← summary (post-playback)
│   │ Phrases Avg Longest Shortest          │   │
│   │  #1 ▰▰▰▰▰▰▰▰▰ 9.4s                    │   │
│   │  #2 ▰▰▰▰▰     5.2s                    │   │
│   │  …                                    │   │
│   │  Detected 3 breaths. That's 1 more    │   │
│   │  than the 2007 DSP baseline (2).      │   │
│   └───────────────────────────────────────┘   │
│                                               │
│   ▾ Analysis & model details                 │  ← collapsible
│                                               │
└───────────────────────────────────────────────┘
```

When the collapsible **Analysis** is open, it expands to show in order:

1. **Log-mel spectrogram** (image fetched from server) with model+baseline breath markers overlaid.
2. **Waveform** (clean canvas, no markers).
3. **Breath probability curve** with horizontal threshold line, model events shaded above, baseline events shaded below.
4. **Model meta** grid (4 cells): architecture, val PR-AUC, calibration ECE, latency.

All four have a vertical playhead line synced with audio playback.

---

## 3. State machine

```
        ┌────────┐
        │  idle  │  ← initial; no audio loaded
        └───┬────┘
            │ tap 🎤
            ▼
        ┌──────────┐
        │ recording │  ← live timer; max 30s; tap stop to end
        └────┬──────┘
             │ stop tapped or 30s hit
             ▼
        ┌────────────┐
        │ analyzing  │  ← POST /process; spinner overlay; ~1–4s
        └────┬───────┘
             │ response ok
             ▼
        ┌───────────┐
        │  ready    │  ← can play, reset, or re-record
        └────┬──────┘
             │ tap ▶
             ▼
        ┌───────────┐
        │  playing  │  ← coaching animations active
        └────┬──────┘
             │ audio.ended
             ▼
        ┌──────────┐
        │  ended   │  ← summary visible; can replay or re-record
        └──────────┘
```

Phase transitions animate the layout with `.spring(response: 0.4, dampingFraction: 0.8)`.

---

## 4. Component specs

### 4.1 Controls row

Three pills, horizontally arranged, top of content area.

- **▶ Play / ⏸ Pause** — primary action. Liquid Glass capsule with the BreathCoach gradient (teal → indigo → magenta) fill. Disabled in `idle`, `recording`, `analyzing`. Toggles between `▶ Play` and `⏸ Pause` text.
- **↺ Reset** — ghost capsule (glass thin material). Resets audio to 0, hides summary, redraws everything at t=0.
- **🎤 Record / ⏹ Stop** — ghost capsule with magenta border. Becomes solid magenta with a pulsing halo while recording. The state text inside ("3.4s") updates every animation frame.

Web reference: lines 185–190 + CSS lines 63–75 of `index.html`.

### 4.2 Stage (side stats + hero ring)

3-column grid; on iPhone collapses to a vertical stack.

**Left — Last phrase**
- Big number with gradient fill (teal→indigo→magenta `LinearGradient` masked over a `Text`).
- "Last phrase" caption below, dim, uppercase, letter-spaced.
- Idle state: shows "—" in faint grey, no gradient.

**Center — Hero ring**
- Circular progress arc, 360pt × 360pt on phone (300pt on smaller screens).
- Track: 14pt-wide circle, `Color.white.opacity(0.07)`.
- Progress arc: 14pt, stroke-rounded, drawn from -π/2, sweep = `min(1, curPhraseDur / 8) * 2π`.
- Arc color is a **LinearGradient** that changes based on coaching state:
  - In a breath: teal → indigo `[#2ee6c8, #6c7bff]`
  - 30+ sec without breath: red → magenta `[#ff5d72, #ff5ea0]` (danger)
  - 12–30 sec: indigo → magenta `[#6c7bff, #ff5ea0]`
  - < 2 sec: amber → orange `[#ffc24b, #ff8d5e]`
  - else (the "good" range): teal → indigo `[#2ee6c8, #6c7bff]`
- A glowing leading dot at the arc's tip (white fill, 14pt blur shadow).
- Subtle radial highlight overlay when in a breath (teal, 22% alpha at center fading out).
- Behind the ring: a soft blurred gradient blob (46pt blur) that breathes in/out on a 5-sec loop — animate scale 0.92 ↔ 1.08, opacity 0.32 ↔ 0.5.

Center of ring (text stack):
- Tiny uppercase caption: "This phrase"
- Big duration: `9.2s` (font system rounded bold, ~80pt)
- Sub line: `singing` / `breathing…` / `press play` / `done`

Web reference: lines 192–212 + draw function lines 351–382 of `index.html`.

**Right — Breaths**
- Big number with gradient fill (same as Last phrase).
- "Breaths" caption.
- When the count increments mid-playback, run a quick scale pulse: scale 1 → 1.22 → 1 over 0.6s with `cubic-bezier(.2, 1.4, .4, 1)` (use SwiftUI `.symbolEffect(.bounce)` or `.transition(.scale)` with custom curve).

### 4.3 Coach banner

A single line of text inside a Liquid Glass card, centered, ~1.25rem font. Color of the card tint depends on the message category:

| State | Tint | Card border |
|---|---|---|
| Idle / carrying line | none (default glass) | `breathInkMuted` |
| `good` | `breathSafe` (#62B6CB) at 8% fill, 50% stroke | matching |
| `warn` (short phrase) | amber #FFC24B at 8% fill | matching |
| `danger` (30+ sec) | red #FF5D72 at 8% fill | matching |
| `breath` (currently in a breath) | teal #2EE6C8 at 10% fill | matching |

**Messages — copy these verbatim** (from web `phraseQuality()`):

| Phrase duration | Message | Category |
|---|---|---|
| In a breath (event live) | `🫁  Breath — phrase reset.` | breath |
| < 1.2s of singing | `Carrying the line…` | none |
| 1.2 – 2s | `Short phrase — try to carry the line a little longer.` | warn |
| 2 – 4s | `Nice — see if you can stretch the next one further.` | warn |
| 4 – 8s | `Comfortable, well-supported phrasing.` | good |
| 8 – 12s | `Strong breath support — lovely long phrase.` | good |
| 12 – 30s | `Impressive control on that long phrase.` | breath |
| 30s+ | `30+ seconds without a breath — ease off before you strain.` | danger |
| `ended` state | `Session complete — see your summary below.` | good |
| `idle` initial | `Press play to begin.` | none |
| recording error | `Recording failed: <error>` | danger |
| mic permission denied | `Microphone permission denied.` | danger |

Web reference: lines 437–445, 461–463 of `index.html`.

### 4.4 Timeline card

A wide Liquid Glass card containing:
- Card head: "Your breathing pattern" left, `0.00 / 12.40 s` right (current/total).
- A canvas-equivalent view, 140pt tall.
- Legend row below: three swatch+label pairs (line 222–226 in web).

The "canvas" in SwiftUI uses `Canvas` view + `GeometryReader`. Render order (back to front):
1. **Breath region rectangles**: for each event in `predicted_events`, draw a vertical rect from t=start_sec to t=end_sec, full height, `rgba(255,94,160,0.13)` (translucent magenta).
2. **Waveform bars**: for each x-pixel, take the max abs sample in that window. Bars are drawn vertically centered. **Played portion** (x ≤ playhead_x) uses a teal→indigo `LinearGradient` along X. **Upcoming** uses `rgba(120,130,156,0.4)`. As playhead moves, more bars flip from grey to gradient — this is the "your line filling in" effect.
3. **Breath markers**: for each event, draw a vertical thin line at the event midpoint (`rgba(255,94,160,0.3)`, 1pt), and a glowing magenta dot near the bottom (radius 4pt, 8pt blur shadow).
4. **Playhead**: a white vertical line at current time, 1.5pt; small white dot at top with a glow.

Web reference: lines 385–406 of `index.html`.

You'll need the raw waveform on-device. Approach: when the server's response arrives, fetch the audio file URL (server can serve it as a static asset; if not, send the user's recorded WAV directly into `AVAudioFile` → read samples). The simplest path is to keep the recorder's source file around and use `AVAudioFile.read(into:)` to grab the float samples for rendering.

### 4.5 Session summary

Card shown only in `ended` phase. Rises in with `.transition(.move(edge: .bottom).combined(with: .opacity))`.

**Stats row** — 4 cells, equal width:
- `Phrases` (count of `phrase_events`)
- `Avg length` (mean of `phrase_events[].duration_sec`)
- `Longest` (max)
- `Shortest` (min)

Numbers use the gradient fill, captions are dim uppercase.

**Phrase list** — one row per phrase event:
- Left: `#1`, `#2`, … in faint mono.
- Middle: a horizontal bar, height 11pt, rounded ends. Width = `(duration / max_duration) * available_width`. Color depends on duration:
  - `< 2s`: amber→orange gradient
  - `≥ 12s`: indigo→magenta gradient
  - else: teal→indigo gradient
- Right: the duration in dim, e.g., `9.4 s`.

**vs-baseline line** — single sentence at the bottom, separated by a top border, comparing `predicted_events.count` to `ruinskiy_events.count`:
- `bh > ru`: `Detected N breaths. That's D more than the classic 2007 DSP baseline (R) — the neural model catches softer inhales.`
- `bh < ru`: `Detected N breaths. The 2007 baseline flagged D more (R).`
- equal: `Detected N breaths. Same count as the 2007 baseline.`

Web reference: lines 466–480 of `index.html`.

### 4.6 Analysis (collapsible)

Use `DisclosureGroup` or a custom expand/collapse with a chevron. Closed by default. When opened, fade-and-slide its body in.

**Section 1 — Log-mel spectrogram**
- Sub-header: `Log-mel spectrogram` left, `40 bands · 10 ms hop` right (both dim, uppercase, tracked).
- Container: 180pt tall, rounded 12pt corners, very dark fill (#0A0A12).
- Image: fetched from the server's `spectrogram_file` URL (e.g. `http://<server>/clips/<id>.png` or directly the path returned). `AsyncImage` with `.resizable().scaledToFill()`.
- Overlay (drawn in a Canvas atop the image):
  - For each `ruinskiy_event`: small magenta rect at the bottom edge (5pt tall, 7pt above bottom).
  - For each `predicted_event`: small indigo rect at the top edge (5pt tall).

**Section 2 — Waveform** (decorative; clean version of timeline waveform)
- 84pt tall canvas.
- Teal→indigo gradient bars, no markers, no playhead in the static draw.

**Section 3 — Breath probability**
- Sub-header: `Breath probability` left, `threshold 0.50` right (the threshold value from response).
- 104pt tall canvas.
- Drawing order:
  - Magenta shading at the bottom (10pt strip) for each `ruinskiy_event`.
  - Indigo shading top-to-bottom (full height) at 26% alpha for each `predicted_event`.
  - Horizontal dashed amber line at y = `H * (1 - threshold)`.
  - Probability curve (teal→indigo gradient, 2pt stroke): for each frame in `breath_prob`, plot `(i/n) * W, H * (1 - prob[i])`.
- Legend row below: 4 swatch+label items (`BreathHead p(breath)`, `threshold`, `detected breath`, `Ruinskiy 2007 baseline`).

**Playhead** in all three of these: a vertical white line at the current time, 1pt, 80% alpha. Re-drawn on each tick.

**Section 4 — Model meta**
- A 4-cell grid (one row on phone is fine):
  - `Architecture` ← `BreathHead · N params` from `model_meta`
  - `Val PR-AUC` ← hardcoded `0.65` for now (until we expose it)
  - `Calibration (ECE)` ← hardcoded `0.022`
  - `Latency` ← `model_meta.per_frame_ms.toFixed(2) ms/frame`
- Each cell is a small rounded inner card with a mono `v` and dim caption `l`.

Web reference: lines 238–267 of `index.html` + draw funcs 408–427.

---

## 5. Data contract

The iOS app calls `POST /process` (multipart with raw WAV) and receives the same JSON the web app uses. Sample shape:

```jsonc
{
  "audio_file": "recording_1234567890.wav",     // string | null — relative path on server
  "spectrogram_file": "recording_1234567890.png",
  "duration_sec": 12.453,
  "sample_rate": 16000,
  "hop_sec": 0.01,
  "n_frames": 1245,
  "breath_prob":  [0.012, 0.014, …],            // length n_frames
  "voiced_prob":  [0.91, 0.92, …],              // length n_frames
  "pitch_norm":   [0.42, 0.43, …],              // length n_frames, 0..1
  "predicted_events": [
    { "start_sec": 1.83, "end_sec": 2.06 },
    …
  ],
  "ruinskiy_events": [
    { "start_sec": 1.81, "end_sec": 2.07, "score": 0.71 },
    …
  ],
  "phrase_events": [
    { "start_sec": 0.0, "end_sec": 4.21, "duration_sec": 4.21 },
    …
  ],
  "threshold": 0.50,
  "phrases_from": "breath_head",                // or "ruinskiy"
  "model_meta": {
    "hidden": 8,
    "params": 15705,
    "inference_ms": 312.4,
    "per_frame_ms": 0.251
  }
}
```

Note: the existing `BreathAPI.swift` has a `DetectResponse` for a thinner `/detect` shape. For Coach you need the *full* `/process` response — define a richer `ProcessResponse` type alongside it. Both endpoints exist on the server.

Endpoint base path: configured in `BreathAPI.baseURL`. The `audio_file` and `spectrogram_file` values are filenames relative to the server's `web/recordings/` (or `clips/`) directory — they're accessible as `\(baseURL)/recordings/<filename>`.

---

## 6. Coaching state computation

The web app computes the current coaching state on every animation frame from the current audio time. Mirror this exactly:

```swift
struct CoachingState {
    var curPhraseDur: Double  // seconds elapsed in the current phrase
    var lastPhraseDur: Double?  // duration of the most recently completed phrase
    var breathCount: Int      // number of breath events whose end_sec <= current time
    var inBreath: Bool        // whether current time is inside any predicted_event
}

func coachingState(at time: Double, response: ProcessResponse) -> CoachingState {
    let events = response.phrasesFrom == .breathHead
        ? response.predictedEvents : response.ruinskiyEvents

    var breathCount = 0
    var inBreath = false
    for ev in events {
        if ev.endSec <= time { breathCount += 1 }
        if ev.startSec <= time && time < ev.endSec { inBreath = true }
    }

    var curPhraseDur = 0.0
    var lastPhraseDur: Double? = nil
    for (i, ph) in response.phraseEvents.enumerated() {
        if ph.startSec <= time && time < ph.endSec {
            curPhraseDur = time - ph.startSec
            if i > 0 { lastPhraseDur = response.phraseEvents[i-1].durationSec }
            break
        } else if time >= ph.endSec {
            lastPhraseDur = ph.durationSec
        }
    }

    return CoachingState(curPhraseDur: curPhraseDur,
                         lastPhraseDur: lastPhraseDur,
                         breathCount: breathCount,
                         inBreath: inBreath)
}
```

Web reference: lines 431–437 of `index.html`.

---

## 7. Animation cadence

Drive the per-frame redraw with `CADisplayLink` (60Hz preferred) while in `playing` phase. On each tick:
- Read `audioPlayer.currentTime`.
- Recompute `coachingState`.
- Trigger SwiftUI updates by writing to `@Observable` state (the ring's `curPhraseDur`, the count, the banner message). Use `Canvas` views that depend on those state vars and they'll redraw automatically.

Stop the link in any non-playing phase.

For phase-driven layout changes (e.g. summary card appearing), use `.animation(.spring(response: 0.4, dampingFraction: 0.8), value: phase)` on the parent VStack.

---

## 8. Liquid Glass translation

The web app uses dark frosted cards over an aurora background. iOS 26 Liquid Glass replaces this naturally:

- **Aurora background** → use `BreathBackdrop` (already in `Components/GlassCard.swift`) — a static linear gradient. Optionally add 2–3 large blurred `Circle()`s animating gentle position drift to mimic the web's aurora.
- **`var(--card)` (rgba(20,24,38,0.66) + blur(12px))** → `GlassCard(cornerRadius: 24)`. Use the existing component for everything.
- **Inner cards in Analysis** (cells in model grid) → smaller glass cards with `cornerRadius: 12`.
- **Pill controls** → `glassEffect(.thin, in: .capsule)` per the design.md doc.
- The **gradient** `linear-gradient(100deg, #2ee6c8 0%, #6c7bff 52%, #ff5ea0 100%)` is the BreathCoach signature. Define it once:
  ```swift
  static let breathGradient = LinearGradient(
    colors: [Color(hex: 0x2EE6C8), Color(hex: 0x6C7BFF), Color(hex: 0xFF5EA0)],
    startPoint: .leading, endPoint: .trailing
  )
  ```
  Use it for: Play button, ring progress arc, ring's leading dot glow, summary stat numbers, phrase bars (default tier), waveform played-portion fill.

Color tokens stay as defined in `GlassCard.swift` extension, but ADD these gradient stops as named statics so the spec is one swap away from being wired:

```swift
extension Color {
    static let breathTeal    = Color(red: 0.184, green: 0.902, blue: 0.784)  // #2EE6C8
    static let breathIndigo  = Color(red: 0.424, green: 0.482, blue: 1.000)  // #6C7BFF
    static let breathMagenta = Color(red: 1.000, green: 0.369, blue: 0.627)  // #FF5EA0
    static let breathAmber   = Color(red: 1.000, green: 0.761, blue: 0.294)  // #FFC24B
    static let breathRed     = Color(red: 1.000, green: 0.365, blue: 0.447)  // #FF5D72
}
```

---

## 9. Out of scope (for this iteration)

- **Mimic mode** — keep `MimicView.swift` exactly as the current placeholder. Don't wire song loading, comparison, feedback. The Home screen still shows the Mimic card; tapping it should push a view that says `Mimic is coming soon.`
- **Real-time streaming inference** — Coach mode here is record → analyze → playback-with-coaching. We are NOT doing chunked POST during recording. The web app doesn't either.
- **Pitch overlay** — `pitch_norm` is in the response but not visualized in the current web Coach. Skip it.
- **VAD overlay** — same, `voiced_prob` is unused in web Coach.
- **Save / share** — recordings stay on-device for the session. No export, no history.

---

## 10. Order of work (suggested)

1. **Wire the `/process` call** — define `ProcessResponse`, hit the endpoint from a tap of the Record button, render nothing yet.
2. **Hero ring** — phrase duration + state-driven color. Use a fake response to drive it first.
3. **Coach banner** — copy the phrase-quality messages verbatim, hook to state.
4. **Side stats** (Last phrase, Breath count) with their pulses.
5. **Timeline card** — waveform from `AVAudioFile` + breath rects + markers + playhead.
6. **Playback loop** — AVAudioPlayer playing the returned recording while a `CADisplayLink` drives state updates.
7. **Session summary** — appears on `ended`, animates in.
8. **Analysis (collapsible)** — spectrogram with overlay, waveform decorative, probability curve, model meta grid.
9. **Polish** — animations, haptics on breath detection, reduced-motion fallback.

Each step is independently testable. Don't refactor as you go.

---

## 11. Reference files

| File | Purpose |
|---|---|
| `../breathcoach/src/nanobreath/deployment/web/index.html` | Web Coach — behavior reference |
| `../breathcoach/src/nanobreath/deployment/serve.py` | Server — endpoint shapes |
| `BreathCoach/Views/Components/GlassCard.swift` | Existing glass primitive + color tokens |
| `BreathCoach/Services/BreathAPI.swift` | Existing API client (extend with `process()` method) |
| `docs/design.md` | Higher-level design principles for the app |
