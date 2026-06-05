//
//  HomeView.swift
//  BreathCoach
//
//  Landing screen. Centered hero (orb + wordmark + tagline), then two
//  identically-structured mode cards stacked below. Mimic stays visually
//  parallel to Coach but isn't tappable — its "SOON" badge does the talking.
//

import SwiftUI

struct HomeView: View {
    var body: some View {
        NavigationStack {
            ZStack {
                BreathBackdrop().ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 28) {
                        hero
                        coachCard
                        mimicCard
                        Spacer(minLength: 12)
                    }
                    .padding(.horizontal, 24)
                    .padding(.top, 32)
                    .padding(.bottom, 24)
                    .frame(maxWidth: .infinity)
                }
                .scrollIndicators(.hidden)
            }
            .navigationDestination(for: AppRoute.self) { route in
                switch route {
                case .coach: CoachView()
                case .mimic: MimicView()
                }
            }
        }
    }

    // MARK: - Hero

    private var hero: some View {
        VStack(spacing: 20) {
            BreathingOrb()
                .frame(width: 104, height: 104)
            wordmark
            tagline
        }
        .frame(maxWidth: .infinity)
        .multilineTextAlignment(.center)
    }

    private var wordmark: some View {
        HStack(spacing: 0) {
            Text("Breath")
                .foregroundStyle(LinearGradient.breathRibbon)
            Text("Coach")
                .foregroundStyle(Color.breathInkPrimary)
        }
        .font(.system(size: 44, weight: .bold, design: .rounded))
        .tracking(-0.5)
    }

    private var tagline: some View {
        Text("Breath & phrase coaching for singers.")
            .font(.system(.callout, design: .rounded))
            .foregroundStyle(Color.breathInkPrimary.opacity(0.7))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 12)
    }

    // MARK: - Cards

    private var coachCard: some View {
        NavigationLink(value: AppRoute.coach) {
            ModeCard(
                icon: "waveform.badge.mic",
                statusText: "LIVE",
                statusTint: .breathMagenta,
                title: "Coach",
                subtitle: "Sing into the mic. See where your breaths land.",
                isActive: true
            )
        }
        .buttonStyle(.plain)
    }

    private var mimicCard: some View {
        // Visually parallel to Coach; intentionally not wrapped in a
        // NavigationLink so the "SOON" state is unambiguous.
        ModeCard(
            icon: "music.note.list",
            statusText: "SOON",
            statusTint: .breathAmber,
            title: "Mimic",
            subtitle: "Pick a reference song, sing along, see how you compare.",
            isActive: false
        )
    }
}

// MARK: - Shared mode card

private struct ModeCard: View {
    let icon: String
    let statusText: String
    let statusTint: Color
    let title: String
    let subtitle: String
    let isActive: Bool

    // Both cards stretch to the same height. The taller subtitle of either
    // card decides — minHeight pins the floor.
    private let minHeight: CGFloat = 200

    var body: some View {
        VStack(spacing: 18) {
            topRow
            VStack(spacing: 8) {
                Text(title)
                    .font(.system(.largeTitle, design: .rounded, weight: .bold))
                    .foregroundStyle(Color.breathInkPrimary)
                Text(subtitle)
                    .font(.system(.callout, design: .rounded))
                    .foregroundStyle(Color.breathInkPrimary.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 6)
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity)
        .frame(minHeight: minHeight)
        .background {
            if #available(iOS 26.0, *) {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .glassEffect(.regular, in: .rect(cornerRadius: 28))
            } else {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .fill(.ultraThinMaterial)
            }
        }
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .strokeBorder(LinearGradient.breathRibbon, lineWidth: 1.5)
                .opacity(0.7)
        }
        .opacity(isActive ? 1 : 0.78)
    }

    private var topRow: some View {
        HStack(alignment: .center) {
            Image(systemName: icon)
                .font(.system(size: 32, weight: .regular))
                .foregroundStyle(LinearGradient.breathRibbon)
                .frame(width: 40, height: 40)
            Spacer()
            statusPill
        }
    }

    private var statusPill: some View {
        Text(statusText)
            .font(.system(.caption2, design: .rounded, weight: .semibold))
            .tracking(2)
            .foregroundStyle(Color.breathInkPrimary.opacity(0.85))
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(Capsule().fill(statusTint.opacity(0.25)))
            .overlay(Capsule().strokeBorder(statusTint.opacity(0.55), lineWidth: 0.8))
    }
}

// MARK: - Breathing orb

private struct BreathingOrb: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        if reduceMotion {
            orb(scale: 1.0, glow: 0.7)
        } else {
            TimelineView(.animation(minimumInterval: 1.0/30.0, paused: false)) { ctx in
                let t = ctx.date.timeIntervalSinceReferenceDate
                // ~5 s breath cycle. Scale + glow ride the same sine so the orb
                // feels like one living thing rather than two layered animations.
                let s = sin(t * 2 * .pi / 5.0)
                orb(scale: 1.0 + s * 0.06, glow: 0.55 + s * 0.25)
            }
        }
    }

    private func orb(scale: Double, glow: Double) -> some View {
        ZStack {
            Circle()
                .fill(LinearGradient.breathRibbon)
                .blur(radius: 24)
                .scaleEffect(scale + 0.18)
                .opacity(glow * 0.85)

            Circle()
                .fill(LinearGradient.breathRibbon)
                .scaleEffect(scale)
                .shadow(color: Color.breathMagenta.opacity(glow), radius: 14)
                .shadow(color: Color.breathIndigo.opacity(glow * 0.6), radius: 22)
        }
    }
}

enum AppRoute: Hashable {
    case coach, mimic
}

#Preview {
    HomeView()
        .environment(ServerHealth())
}
