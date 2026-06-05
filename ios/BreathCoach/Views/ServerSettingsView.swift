//
//  ServerSettingsView.swift
//  BreathCoach
//
//  Lets the user point the app at their Mac's model server, and surfaces a
//  live connection status. Replaces the old hardcoded IP so a Wi-Fi change no
//  longer means a rebuild.
//

import SwiftUI

/// Compact connection indicator for a nav-bar toolbar. Tapping opens Settings.
struct ServerStatusButton: View {
    @Environment(ServerHealth.self) private var serverHealth
    @State private var showingSettings = false

    var body: some View {
        Button { showingSettings = true } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(style.color)
                    .frame(width: 8, height: 8)
                    .shadow(color: style.color.opacity(0.7), radius: 3)
                Text(style.label)
                    .font(.system(.caption, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary.opacity(0.75))
            }
        }
        .accessibilityLabel("Server \(style.label). Open settings.")
        .sheet(isPresented: $showingSettings) {
            ServerSettingsView()
        }
    }

    private var style: (color: Color, label: String) {
        switch serverHealth.status {
        case .reachable:   return (.green, "Connected")
        case .unreachable: return (.breathRed, "Offline")
        case .checking:    return (.breathAmber, "Checking…")
        case .unknown:     return (.breathDim, "Set up")
        }
    }
}

struct ServerSettingsView: View {
    @Environment(ServerHealth.self) private var serverHealth
    @Environment(\.dismiss) private var dismiss

    @State private var draft = ServerStore.load()
    @State private var testing = false

    var body: some View {
        NavigationStack {
            ZStack {
                BreathBackdrop().ignoresSafeArea()
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        GlassCard {
                            VStack(alignment: .leading, spacing: 16) {
                                addressField
                                Divider().opacity(0.4)
                                statusRow
                            }
                        }
                        actions
                        hint
                    }
                    .padding(20)
                }
                .scrollIndicators(.hidden)
            }
            .navigationTitle("Model server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") { save() }
                        .fontWeight(.semibold)
                        .disabled(!isValid)
                }
            }
        }
    }

    // MARK: - Pieces

    private var addressField: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("SERVER ADDRESS")
                .font(.system(.caption2, design: .rounded, weight: .semibold))
                .tracking(1.5)
                .foregroundStyle(Color.breathInkMuted)
            TextField("192.168.1.5:8080", text: $draft)
                .textFieldStyle(.plain)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(Color.breathInkPrimary)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .submitLabel(.done)
                .onSubmit { if isValid { save() } }
                .padding(.horizontal, 14).padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.white.opacity(0.10))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(isValid ? Color.white.opacity(0.18)
                                              : Color.breathRed.opacity(0.6),
                                      lineWidth: 1)
                )
            if !isValid {
                Text("Enter a host and port, e.g. 192.168.1.5:8080")
                    .font(.system(.caption2, design: .rounded))
                    .foregroundStyle(Color.breathRed)
            }
        }
    }

    private var statusRow: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)
                .shadow(color: statusColor.opacity(0.7), radius: 4)
            VStack(alignment: .leading, spacing: 2) {
                Text(statusText)
                    .font(.system(.callout, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary)
                if let version = serverHealth.modelVersion, serverHealth.isReachable {
                    Text("Model \(version)")
                        .font(.system(.caption, design: .rounded))
                        .foregroundStyle(Color.breathInkMuted)
                } else if let error = serverHealth.lastError, serverHealth.status == .unreachable {
                    Text(error)
                        .font(.system(.caption, design: .rounded))
                        .foregroundStyle(Color.breathInkMuted)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 0)
        }
    }

    private var actions: some View {
        Button { test() } label: {
            HStack(spacing: 8) {
                if testing { ProgressView().tint(.white) }
                else { Image(systemName: "antenna.radiowaves.left.and.right") }
                Text(testing ? "Testing…" : "Test connection")
            }
            .font(.system(.callout, design: .rounded, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Capsule().fill(LinearGradient.breathRibbon).opacity(isValid ? 1 : 0.35))
        }
        .disabled(!isValid || testing)
    }

    private var hint: some View {
        Text("Find your Mac's address by running `ipconfig getifaddr en0` in Terminal, then keep the `:8080` port. Both devices must be on the same Wi-Fi.")
            .font(.system(.caption, design: .rounded))
            .foregroundStyle(Color.breathInkPrimary.opacity(0.65))
            .padding(.horizontal, 4)
    }

    // MARK: - Logic

    private var isValid: Bool { ServerStore.url(from: draft) != nil }

    private func save() {
        ServerStore.save(draft)
        Task { await serverHealth.check() }
        dismiss()
    }

    private func test() {
        ServerStore.save(draft)
        testing = true
        Task {
            await serverHealth.check()
            testing = false
        }
    }

    private var statusColor: Color {
        switch serverHealth.status {
        case .reachable:   return .green
        case .unreachable: return .breathRed
        case .checking:    return .breathAmber
        case .unknown:     return .breathDim
        }
    }

    private var statusText: String {
        switch serverHealth.status {
        case .reachable:   return "Connected"
        case .unreachable: return "Not reachable"
        case .checking:    return "Checking…"
        case .unknown:     return "Not tested yet"
        }
    }
}
