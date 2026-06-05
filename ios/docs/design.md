# Design rationale

## Principles

1. **Calm before clever.** Singing is emotionally exposing — the UI should feel like a friend listening, not a coach with a clipboard. No red error states. No grade letters. Feedback uses neutral verbs ("noticed", "compared", "matched").
2. **One thing per screen.** Home picks a mode. Coach shows your breaths. Mimic compares. Feedback explains. Each screen has one job.
3. **Liquid Glass is the visual language.** Translucent layered surfaces over a single sage→ocean gradient backdrop. Nothing else competes for attention.
4. **Touch targets ≥ 48pt.** Phone in hand, mic on, you don't tap a 24pt button.
5. **No sign-in.** Local-first. The professor opens the app and it works. No "create account" wall.

## Color tokens

| Token | Hex | Use |
|---|---|---|
| `breathAccent` | `#5BC0BE` | Primary accent — buttons, active states |
| `breathInkPrimary` | `#0B132B` | Body text on light glass |
| `breathInkMuted` | `#1C2541` | Secondary text |
| `breathSurface` | `#FFFFFF` @ 0.6α | Card body (Liquid Glass infers the rest) |
| `breathSafe` | `#62B6CB` | "Good" comparison cells in Mimic feedback |
| `breathFlag` | `#E0B354` | "Noticed" cells (no judgment) |
| `breathBackdrop` | linear-gradient(155°, #B6DCE3 0%, #5BC0BE 50%, #1C5D7E 100%) | Backdrop |

Dark mode flips ink colors but keeps the gradient (just darker stops).

## Typography

- SF Pro Display, Rounded (system) — bold for headings, regular for body
- One scale: 32 / 22 / 17 / 13. Nothing else.
- Never center body text. Always left-align.

## Motion

- Breath marker entry: spring(response: 0.4, dampingFraction: 0.7), scale 0.6→1.0, opacity 0→1
- Screen transitions: standard SwiftUI .push, no custom transitions
- Audio playback head: 60Hz update, hardware-synced via CADisplayLink

## Screen flow

```
HomeView
├─ "Coach" → CoachView (live mic) → SessionSummary
└─ "Mimic" → SongPickerView → MimicView (play+sing) → FeedbackView
```

Back stack respects iOS conventions (swipe from edge dismisses).

## What the UI deliberately doesn't show

- Numerical F1 / precision / recall — meaningless to a singer
- Model confidence per detection — internal detail
- Frame-by-frame visualizations — too dense
- A "your score" headline number — feels like grading

Instead, the UI shows:
- Counts ("you breathed 6 times")
- Comparisons ("the original singer breathed 8 times, here's where")
- Suggestions, optionally ("phrase 3 might be easier with a breath after 'midnight'")
