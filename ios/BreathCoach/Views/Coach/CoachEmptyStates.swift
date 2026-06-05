//
//  CoachEmptyStates.swift
//  BreathCoach
//
//  The idle invitation and the analyzing spinner — what the Coach screen shows
//  before there's a recording to play back.
//

import SwiftUI

/// Shown in the `idle` phase: an inviting prompt to record.
struct IdlePromptView: View {
    var body: some View {
        GlassCard {
            VStack(spacing: 14) {
                Image(systemName: "waveform.badge.mic")
                    .font(.system(size: 46, weight: .regular))
                    .foregroundStyle(LinearGradient.breathRibbon)
                Text("Sing a phrase")
                    .font(.system(.title2, design: .rounded, weight: .bold))
                    .foregroundStyle(Color.breathInkPrimary)
                Text("Tap Record and sing for 5–30 seconds. I'll play it back and show you where you breathe.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(Color.breathInkPrimary.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 28)
        }
        .accessibilityElement(children: .combine)
    }
}

/// Shown in the `analyzing` phase: the model is running on the Mac.
struct AnalyzingView: View {
    var body: some View {
        GlassCard {
            VStack(spacing: 14) {
                ProgressView()
                    .scaleEffect(1.2)
                    .tint(.breathIndigo)
                Text("Reading your phrasing…")
                    .font(.system(.title3, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary)
                Text("Finding your breaths and mapping each phrase.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(Color.breathInkPrimary.opacity(0.7))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 28)
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    VStack(spacing: 20) {
        IdlePromptView()
        AnalyzingView()
    }
    .padding(20)
    .background(BreathBackdrop())
}
