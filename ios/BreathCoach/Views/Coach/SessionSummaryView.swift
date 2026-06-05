//
//  SessionSummaryView.swift
//  BreathCoach
//
//  Appears after playback ends. Stats row + per-phrase bars + a one-liner
//  comparing the model's breath count to the 2007 DSP baseline.
//  See ui.md §4.5.
//

import SwiftUI

struct SessionSummaryView: View {
    let response: ProcessResponse

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 16) {
                Text("Session summary")
                    .font(.system(.title3, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary)

                statsRow

                if !response.phraseEvents.isEmpty {
                    Divider().opacity(0.4)
                    phraseList
                }

                Divider().opacity(0.4)
                vsBaselineLine
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(Color.breathInkPrimary.opacity(0.8))
            }
        }
    }

    // MARK: - Stats row

    private var statsRow: some View {
        let durations = response.phraseEvents.map(\.durationSec)
        let count = response.phraseEvents.count
        let avg = durations.isEmpty ? 0 : durations.reduce(0, +) / Double(durations.count)
        let longest = durations.max() ?? 0
        let shortest = durations.min() ?? 0
        return HStack(spacing: 0) {
            statCell(value: "\(count)", label: "Phrases")
            statCell(value: String(format: "%.1fs", avg), label: "Avg")
            statCell(value: String(format: "%.1fs", longest), label: "Longest")
            statCell(value: String(format: "%.1fs", shortest), label: "Shortest")
        }
    }

    private func statCell(value: String, label: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(.title2, design: .rounded, weight: .bold))
                .foregroundStyle(LinearGradient.breathRibbon)
            Text(label.uppercased())
                .font(.system(.caption2, design: .rounded, weight: .semibold))
                .tracking(1.5)
                .foregroundStyle(Color.breathInkMuted)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Phrase list

    private var phraseList: some View {
        let maxDuration = max(response.phraseEvents.map(\.durationSec).max() ?? 1, 1)
        return VStack(spacing: 8) {
            ForEach(Array(response.phraseEvents.enumerated()), id: \.offset) { idx, phrase in
                phraseRow(index: idx + 1, phrase: phrase, maxDuration: maxDuration)
            }
        }
    }

    private func phraseRow(index: Int, phrase: PhraseEvent, maxDuration: Double) -> some View {
        HStack(spacing: 12) {
            Text(String(format: "#%d", index))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(Color.breathInkMuted)
                .frame(width: 28, alignment: .leading)

            GeometryReader { geo in
                let fraction = phrase.durationSec / maxDuration
                Capsule()
                    .fill(barGradient(for: phrase.durationSec))
                    .frame(width: max(8, geo.size.width * fraction), height: 11)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 11)

            Text(String(format: "%.1fs", phrase.durationSec))
                .font(.system(.caption, design: .rounded).monospacedDigit())
                .foregroundStyle(Color.breathInkMuted)
                .frame(width: 44, alignment: .trailing)
        }
    }

    private func barGradient(for dur: Double) -> LinearGradient {
        if dur < 2  { return .breathWarn }
        if dur >= 12 { return LinearGradient(colors: [.breathIndigo, .breathMagenta],
                                             startPoint: .leading, endPoint: .trailing) }
        return .breathRibbon
    }

    // MARK: - vs baseline copy

    @ViewBuilder private var vsBaselineLine: some View {
        let predicted = response.predictedEvents.count
        let ruinskiy = response.ruinskiyEvents.count
        if response.phrasesFrom == "ruinskiy" {
            // Breaths were counted from the DSP baseline this session, so the
            // headline matches what played back; note what the neural model saw.
            Text("Detected \(ruinskiy) breaths with the classic 2007 DSP baseline. The neural model flagged \(predicted) this time.")
        } else {
            let diff = predicted - ruinskiy
            if diff > 0 {
                Text("Detected \(predicted) breaths. That's \(diff) more than the classic 2007 DSP baseline (\(ruinskiy)) — the neural model catches softer inhales.")
            } else if diff < 0 {
                Text("Detected \(predicted) breaths. The 2007 baseline flagged \(-diff) more (\(ruinskiy)).")
            } else {
                Text("Detected \(predicted) breaths. Same count as the 2007 baseline.")
            }
        }
    }
}
