//
//  CoachBannerView.swift
//  BreathCoach
//
//  A single calm line of guidance, tinted by the current coaching tone.
//

import SwiftUI

struct CoachBannerView: View {
    let message: CoachingMessage

    var body: some View {
        HStack(spacing: 12) {
            if let icon = leadingIcon {
                Image(systemName: icon)
                    .foregroundStyle(iconColor)
                    .font(.system(.body, weight: .semibold))
                    .accessibilityHidden(true)
            }
            Text(message.text)
                .font(.system(.callout, design: .rounded, weight: .medium))
                .foregroundStyle(Color.breathInkPrimary)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 14)
        .background {
            if #available(iOS 26.0, *) {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .glassEffect(.regular, in: .rect(cornerRadius: 18))
            } else {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.ultraThinMaterial)
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(toneFill)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(toneStroke, lineWidth: 1)
        }
        .animation(.easeInOut(duration: 0.25), value: message.tone)
    }

    // MARK: - Tone styling

    private var leadingIcon: String? {
        switch message.tone {
        case .breath: return "wind"
        case .good:   return "checkmark.seal.fill"
        case .warn:   return "leaf.fill"
        case .danger: return "exclamationmark.triangle.fill"
        case .neutral: return nil
        }
    }

    private var iconColor: Color {
        switch message.tone {
        case .breath: return .breathTeal
        case .good:   return .breathSafe
        case .warn:   return .breathAmber
        case .danger: return .breathRed
        case .neutral: return .breathInkMuted
        }
    }

    private var toneFill: Color {
        switch message.tone {
        case .breath: return Color.breathTeal.opacity(0.10)
        case .good:   return Color.breathSafe.opacity(0.08)
        case .warn:   return Color.breathAmber.opacity(0.08)
        case .danger: return Color.breathRed.opacity(0.10)
        case .neutral: return Color.clear
        }
    }

    private var toneStroke: Color {
        switch message.tone {
        case .breath: return Color.breathTeal.opacity(0.45)
        case .good:   return Color.breathSafe.opacity(0.45)
        case .warn:   return Color.breathAmber.opacity(0.45)
        case .danger: return Color.breathRed.opacity(0.45)
        case .neutral: return Color.white.opacity(0.18)
        }
    }
}
