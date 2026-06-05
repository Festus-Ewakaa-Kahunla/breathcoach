//
//  MimicView.swift
//  BreathCoach
//
//  Placeholder per ui.md §9 — Mimic mode is "coming soon" for this iteration.
//  Wiring will arrive once the Coach view is locked in.
//

import SwiftUI

struct MimicView: View {
    var body: some View {
        ZStack {
            BreathBackdrop().ignoresSafeArea()

            VStack(spacing: 20) {
                Spacer()
                Image(systemName: "music.note.list")
                    .font(.system(size: 56, weight: .light))
                    .foregroundStyle(LinearGradient.breathRibbon)
                Text("Mimic is coming soon")
                    .font(.system(.title2, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary)
                Text("Pick a reference song, sing along, and see how your breath patterns compare. Coming after Coach is locked in.")
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(Color.breathInkMuted)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 28)
                Spacer()
            }
        }
        .navigationTitle("Mimic")
        .navigationBarTitleDisplayMode(.inline)
    }
}

#Preview {
    NavigationStack { MimicView() }
}
