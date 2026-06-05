//
//  TimelineCardView.swift
//  BreathCoach
//
//  The "your breathing pattern" card. Renders the recorded waveform with
//  breath-region overlays, magenta breath markers, and a synced playhead.
//  See ui.md §4.4.
//

import SwiftUI

struct TimelineCardView: View {
    let peaks: [Float]
    let events: [BreathEvent]   // predicted_events from /process
    let duration: Double
    let currentTime: Double
    let isInteractive: Bool     // dims when no audio is loaded
    var onSeek: (Double) -> Void = { _ in }

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                head
                GeometryReader { geo in
                    Canvas { context, size in
                        drawBreathRegions(in: context, size: size)
                        drawWaveform(in: context, size: size)
                        drawBreathMarkers(in: context, size: size)
                        drawPlayhead(in: context, size: size)
                    }
                    .contentShape(Rectangle())
                    .gesture(scrubGesture(width: geo.size.width))
                }
                .frame(height: 220)
                legend
            }
            .opacity(isInteractive ? 1 : 0.5)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Breathing pattern")
            .accessibilityValue(accessibilityValue)
            .accessibilityAdjustableAction { direction in
                guard isInteractive, duration > 0 else { return }
                switch direction {
                case .increment: onSeek(min(duration, currentTime + 1))
                case .decrement: onSeek(max(0, currentTime - 1))
                @unknown default: break
                }
            }
        }
    }

    private var accessibilityValue: String {
        String(format: "%d breaths. %.0f of %.0f seconds.",
                events.count, currentTime, duration)
    }

    /// Tap or drag anywhere on the waveform to move the playhead. A zero-distance
    /// drag fires immediately, so a plain tap seeks too.
    private func scrubGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard isInteractive, duration > 0, width > 0 else { return }
                let fraction = min(max(0, value.location.x / width), 1)
                onSeek(fraction * duration)
            }
    }

    // MARK: - Card chrome

    private var head: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("BREATHING PATTERN")
                    .font(.system(.caption, design: .rounded, weight: .semibold))
                    .tracking(2)
                    .foregroundStyle(Color.breathMagenta)
                Text(timeLabel)
                    .font(.system(size: 32, weight: .bold, design: .rounded).monospacedDigit())
                    .foregroundStyle(LinearGradient.breathRibbon)
                    .contentTransition(.numericText())
            }
            Spacer()
        }
    }

    private var timeLabel: String {
        String(format: "%.2f / %.2f s", currentTime, duration)
    }

    private var legend: some View {
        HStack(spacing: 14) {
            legendSwatch(LinearGradient.breathRibbon, "Your voice")
            legendDot(.breathMagenta, "Breath")
            legendBlock(Color.breathMagenta.opacity(0.28), "Breath region")
            Spacer(minLength: 0)
        }
        .font(.system(.caption2, design: .rounded))
        .foregroundStyle(Color.breathInkMuted)
    }

    private func legendSwatch(_ gradient: LinearGradient, _ label: String) -> some View {
        HStack(spacing: 6) {
            Capsule().fill(gradient).frame(width: 18, height: 6)
            Text(label)
        }
    }

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
        }
    }

    private func legendBlock(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 18, height: 8)
            Text(label)
        }
    }

    // MARK: - Drawing

    private func drawBreathRegions(in context: GraphicsContext, size: CGSize) {
        guard duration > 0 else { return }
        let fill = Color.breathMagenta.opacity(0.13)
        for event in events {
            let x0 = CGFloat(event.startSec / duration) * size.width
            let x1 = CGFloat(event.endSec / duration) * size.width
            let rect = CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: size.height)
            context.fill(Path(rect), with: .color(fill))
        }
    }

    private func drawWaveform(in context: GraphicsContext, size: CGSize) {
        guard !peaks.isEmpty else { return }
        let mid = size.height / 2
        let playheadX = duration > 0 ? CGFloat(currentTime / duration) * size.width : 0
        let bucket = size.width / CGFloat(peaks.count)
        let played = GraphicsContext.Shading.linearGradient(
            Gradient(colors: [.breathTeal, .breathIndigo, .breathMagenta]),
            startPoint: .zero, endPoint: CGPoint(x: size.width, y: 0)
        )
        // Upcoming bars are notably dimmer so the played/unplayed split reads
        // at a glance, not just on close inspection.
        let upcoming = GraphicsContext.Shading.color(Color.white.opacity(0.18))

        for (i, peak) in peaks.enumerated() {
            let x = CGFloat(i) * bucket
            let halfHeight = max(1, CGFloat(peak) * (size.height * 0.46))
            let rect = CGRect(x: x, y: mid - halfHeight,
                              width: max(1, bucket * 0.8),
                              height: halfHeight * 2)
            let path = Path(roundedRect: rect, cornerRadius: max(1, bucket * 0.4))
            context.fill(path, with: x <= playheadX ? played : upcoming)
        }
    }

    private func drawBreathMarkers(in context: GraphicsContext, size: CGSize) {
        guard duration > 0 else { return }
        for event in events {
            let mid = CGFloat((event.startSec + event.endSec) / 2 / duration) * size.width
            // Vertical thin line through the marker
            var line = Path()
            line.move(to: CGPoint(x: mid, y: 0))
            line.addLine(to: CGPoint(x: mid, y: size.height))
            context.stroke(line, with: .color(Color.breathMagenta.opacity(0.3)), lineWidth: 1)

            // Glowing dot at the bottom
            let dotRadius: CGFloat = 4
            let dotRect = CGRect(x: mid - dotRadius,
                                 y: size.height - dotRadius * 2 - 2,
                                 width: dotRadius * 2,
                                 height: dotRadius * 2)
            context.drawLayer { layer in
                layer.addFilter(.blur(radius: 6))
                layer.fill(Path(ellipseIn: dotRect),
                           with: .color(Color.breathMagenta.opacity(0.6)))
            }
            context.fill(Path(ellipseIn: dotRect), with: .color(.breathMagenta))
        }
    }

    private func drawPlayhead(in context: GraphicsContext, size: CGSize) {
        guard duration > 0, isInteractive else { return }
        let x = CGFloat(currentTime / duration) * size.width

        // Glow stroke behind the main line so the playhead pops against busy
        // waveform content. Drawn first.
        var glow = Path()
        glow.move(to: CGPoint(x: x, y: -4))
        glow.addLine(to: CGPoint(x: x, y: size.height + 4))
        context.drawLayer { layer in
            layer.addFilter(.blur(radius: 4))
            layer.stroke(glow,
                         with: .color(Color.breathMagenta.opacity(0.85)),
                         lineWidth: 5)
        }

        // Main playhead line — 3pt, bright white.
        var line = Path()
        line.move(to: CGPoint(x: x, y: 0))
        line.addLine(to: CGPoint(x: x, y: size.height))
        context.stroke(line, with: .color(.white), lineWidth: 3)

        // Anchor dots top + bottom so the eye locks onto them as they slide.
        let topDot = CGRect(x: x - 7, y: -7, width: 14, height: 14)
        context.drawLayer { layer in
            layer.addFilter(.blur(radius: 6))
            layer.fill(Path(ellipseIn: topDot),
                       with: .color(Color.breathMagenta.opacity(0.75)))
        }
        context.fill(Path(ellipseIn: topDot), with: .color(.white))

        let bottomDot = CGRect(x: x - 4, y: size.height - 4, width: 8, height: 8)
        context.fill(Path(ellipseIn: bottomDot), with: .color(.white))
    }
}
