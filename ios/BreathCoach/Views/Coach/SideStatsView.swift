//
//  SideStatsView.swift
//  BreathCoach
//
//  Last-phrase and breath-count pills under the hero ring.
//

import SwiftUI

struct SideStatsView: View {
    let state: CoachingState
    let phase: CoachPhase

    var body: some View {
        HStack(spacing: 12) {
            StatCell(
                value: state.lastPhraseDur.map { String(format: "%.1fs", $0) } ?? "—",
                label: "Last phrase",
                isActive: state.lastPhraseDur != nil
            )
            StatCell(
                value: "\(state.breathCount)",
                label: "Breaths",
                isActive: (phase == .playing || phase == .ended) && state.breathCount > 0
            )
        }
        .frame(maxWidth: .infinity)
    }
}

private struct StatCell: View {
    let value: String
    let label: String
    let isActive: Bool

    var body: some View {
        VStack(spacing: 6) {
            // The number sits in its own block so concrete view types
            // (no AnyView) let SwiftUI diff cleanly across phase changes.
            number
            Text(label.uppercased())
                .font(.system(.caption2, design: .rounded, weight: .semibold))
                .tracking(1.5)
                .foregroundStyle(Color.breathInkPrimary.opacity(0.6))
        }
        .frame(maxWidth: .infinity, minHeight: 86)
        .padding(.vertical, 14)
        .padding(.horizontal, 12)
        .background {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(.ultraThinMaterial)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.white.opacity(0.22), lineWidth: 0.5)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(label)
        .accessibilityValue(value)
    }

    @ViewBuilder private var number: some View {
        if isActive {
            Text(value)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(LinearGradient.breathRibbon)
                .contentTransition(.numericText())
        } else {
            Text(value)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundStyle(Color.breathInkPrimary.opacity(0.55))
                .contentTransition(.numericText())
        }
    }
}

#Preview {
    VStack(spacing: 12) {
        SideStatsView(state: .empty, phase: .idle)
        SideStatsView(state: CoachingState(curPhraseDur: 4, lastPhraseDur: 8.1, breathCount: 3, inBreath: false),
                      phase: .playing)
    }
    .padding(20)
    .background(BreathBackdrop())
}
