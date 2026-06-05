//
//  RecordingWaveformView.swift
//  BreathCoach
//
//  The hero of Coach mode while the mic is open. Renders a scrolling live
//  waveform from AudioRecorder.recentLevels, a big elapsed-seconds readout, and
//  a slim progress bar counting toward the recording cap.
//

import SwiftUI

struct RecordingWaveformView: View {
    /// Live peaks from the recorder (one per ~2.5 ms chunk). Newest at the END.
    let levels: [Float]

    /// Capacity the recorder targets — bar pitch stays stable as the buffer
    /// fills and then scrolls.
    let capacity: Int

    /// Newest single peak (0..1) — drives REC dot intensity.
    let currentLevel: Float

    /// Reference for elapsed display. Use Date() at recording start.
    let startedAt: Date?

    /// Recordings auto-stop here; the cap bar fills toward it.
    var maxSeconds: Double = 30

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 18) {
                // One timer drives both the elapsed readout and the cap bar.
                TimelineView(.periodic(from: .now, by: 0.05)) { ctx in
                    let elapsed = elapsedSeconds(now: ctx.date)
                    VStack(alignment: .leading, spacing: 12) {
                        header(elapsed: elapsed)
                        capBar(elapsed: elapsed)
                    }
                }
                Canvas { context, size in
                    drawWaveform(in: context, size: size)
                    drawCenterAxis(in: context, size: size)
                }
                .frame(height: 220)
                hint
            }
        }
    }

    // MARK: - Header

    private func header(elapsed: Double) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            recDot
            VStack(alignment: .leading, spacing: 2) {
                Text("RECORDING")
                    .font(.system(.caption, design: .rounded, weight: .semibold))
                    .tracking(2)
                    .foregroundStyle(Color.breathMagenta)
                Text(String(format: "%.1fs", elapsed))
                    .font(.system(size: 44, weight: .bold, design: .rounded).monospacedDigit())
                    .foregroundStyle(LinearGradient.breathRibbon)
                    .contentTransition(.numericText())
            }
            Spacer(minLength: 0)
        }
    }

    private var recDot: some View {
        // Dot pulses with the mic level so it really feels live.
        let intensity = 0.4 + Double(currentLevel) * 0.6
        return Circle()
            .fill(Color.breathMagenta)
            .frame(width: 14, height: 14)
            .opacity(intensity)
            .shadow(color: Color.breathMagenta.opacity(intensity), radius: 10)
            .animation(.easeOut(duration: 0.08), value: intensity)
    }

    // MARK: - Recording cap

    private func capBar(elapsed: Double) -> some View {
        let fraction = min(1, max(0, elapsed / maxSeconds))
        let remaining = max(0, maxSeconds - elapsed)
        let nearEnd = remaining <= 5
        return VStack(alignment: .leading, spacing: 4) {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.12))
                    Capsule()
                        .fill(nearEnd ? LinearGradient.breathWarn : LinearGradient.breathRibbon)
                        .frame(width: max(2, geo.size.width * fraction))
                }
            }
            .frame(height: 4)
            Text(nearEnd
                 ? String(format: "Wrapping up — %.0fs left", remaining)
                 : String(format: "Up to %.0fs", maxSeconds))
                .font(.system(.caption2, design: .rounded, weight: .medium))
                .foregroundStyle(nearEnd ? Color.breathAmber : Color.breathInkMuted)
                .contentTransition(.numericText())
        }
    }

    private func elapsedSeconds(now: Date) -> Double {
        guard let start = startedAt else { return 0 }
        return max(0, now.timeIntervalSince(start))
    }

    // MARK: - Drawing

    /// Dense vertical bars from min to max amplitude per peak. With ~376 peaks
    /// per second over 720 capacity (~1.9 s history), each bar is ~0.5pt wide
    /// on iPhone — looks like real audio, not a level meter.
    private func drawWaveform(in context: GraphicsContext, size: CGSize) {
        guard !levels.isEmpty, capacity > 0 else { return }
        let mid = size.height / 2
        let barWidth = size.width / CGFloat(capacity)
        let halfMax = size.height * 0.46

        let shading = GraphicsContext.Shading.linearGradient(
            Gradient(colors: [.breathTeal, .breathIndigo, .breathMagenta]),
            startPoint: .zero,
            endPoint: CGPoint(x: size.width, y: 0)
        )

        // Right-anchor: newest peak at the right edge, history trails left.
        let startIndex = capacity - levels.count
        var path = Path()
        for (i, level) in levels.enumerated() {
            let x = (CGFloat(startIndex + i) + 0.5) * barWidth
            // Floor visible amplitude so silent moments still draw a hairline,
            // which keeps the wave visually continuous.
            let half = max(0.5, CGFloat(level) * halfMax)
            path.move(to: CGPoint(x: x, y: mid - half))
            path.addLine(to: CGPoint(x: x, y: mid + half))
        }
        // Stroke width slightly wider than bar pitch so adjacent strokes
        // touch and the wave reads as a solid envelope.
        context.stroke(path, with: shading, lineWidth: max(0.6, barWidth * 1.2))
    }

    private func drawCenterAxis(in context: GraphicsContext, size: CGSize) {
        var axis = Path()
        axis.move(to: CGPoint(x: 0, y: size.height / 2))
        axis.addLine(to: CGPoint(x: size.width, y: size.height / 2))
        context.stroke(axis,
                       with: .color(Color.white.opacity(0.10)),
                       style: StrokeStyle(lineWidth: 0.5))
    }

    // MARK: - Footer

    private var hint: some View {
        Text("Sing into the mic. Tap stop when you're done.")
            .font(.system(.caption, design: .rounded))
            .foregroundStyle(Color.breathInkMuted)
    }
}

#Preview {
    RecordingWaveformView(
        levels: (0..<360).map { i in
            Float(abs(sin(Double(i) * 0.07)) * 0.6 + abs(sin(Double(i) * 0.23)) * 0.3)
        },
        capacity: 720,
        currentLevel: 0.7,
        startedAt: Date().addingTimeInterval(-26.2)
    )
    .padding(20)
    .background(BreathBackdrop())
}
