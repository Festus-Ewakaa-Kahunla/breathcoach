//
//  GlassCard.swift
//  BreathCoach
//
//  Reusable Liquid Glass container. Wraps the iOS 26 `glassEffect` modifier
//  with a fallback for earlier OS versions.
//

import SwiftUI

struct GlassCard<Content: View>: View {
    let cornerRadius: CGFloat
    @ViewBuilder var content: Content

    init(cornerRadius: CGFloat = 24, @ViewBuilder content: () -> Content) {
        self.cornerRadius = cornerRadius
        self.content = content()
    }

    var body: some View {
        content
            .padding(20)
            .background {
                if #available(iOS 26.0, *) {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .glassEffect(.regular, in: .rect(cornerRadius: cornerRadius))
                } else {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .fill(.ultraThinMaterial)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.18), lineWidth: 0.5)
            }
    }
}

struct BreathBackdrop: View {
    var body: some View {
        LinearGradient(
            colors: [
                Color(red: 0.71, green: 0.86, blue: 0.89),    // #B6DCE3
                Color(red: 0.35, green: 0.75, blue: 0.74),    // #5BC0BE
                Color(red: 0.11, green: 0.36, blue: 0.49)     // #1C5D7E
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}

// MARK: - Color tokens
extension Color {
    static let breathAccent     = Color(red: 0.357, green: 0.753, blue: 0.745)
    static let breathInkPrimary = Color(red: 0.043, green: 0.075, blue: 0.169)
    static let breathInkMuted   = Color(red: 0.110, green: 0.145, blue: 0.255)
    static let breathSafe       = Color(red: 0.384, green: 0.714, blue: 0.792)
    static let breathFlag       = Color(red: 0.878, green: 0.702, blue: 0.329)

    // BreathCoach signature palette — matches the web app's CSS variables.
    static let breathTeal       = Color(red: 0.184, green: 0.902, blue: 0.784) // #2EE6C8
    static let breathIndigo     = Color(red: 0.424, green: 0.482, blue: 1.000) // #6C7BFF
    static let breathMagenta    = Color(red: 1.000, green: 0.369, blue: 0.627) // #FF5EA0
    static let breathAmber      = Color(red: 1.000, green: 0.761, blue: 0.294) // #FFC24B
    static let breathRed        = Color(red: 1.000, green: 0.365, blue: 0.447) // #FF5D72
    static let breathInkOnDark  = Color(red: 0.961, green: 0.961, blue: 0.988) // #F5F5FC
    static let breathDim        = Color(red: 0.604, green: 0.631, blue: 0.722) // #9AA1B8
    static let breathFaint      = Color(red: 0.337, green: 0.365, blue: 0.463) // #565D76
}

// MARK: - Signature gradients
extension LinearGradient {
    /// The BreathCoach "ribbon" — teal → indigo → magenta. Used for the Play
    /// button, ring arc, summary numbers, and any moment that should sing.
    static let breathRibbon = LinearGradient(
        colors: [.breathTeal, .breathIndigo, .breathMagenta],
        startPoint: .leading, endPoint: .trailing
    )

    static let breathDanger = LinearGradient(
        colors: [.breathRed, .breathMagenta],
        startPoint: .leading, endPoint: .trailing
    )

    static let breathWarn = LinearGradient(
        colors: [.breathAmber, .breathRed.opacity(0.85)],
        startPoint: .leading, endPoint: .trailing
    )
}
