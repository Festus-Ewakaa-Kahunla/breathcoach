//
//  CoachView.swift
//  BreathCoach
//
//  Live breath coaching screen. Owns the CoachController; the controller owns
//  audio and the network. The TimelineView re-renders the layout at ~60Hz
//  while playing so the ring, banner, and timeline stay in lockstep with
//  audio.currentTime.
//

import SwiftUI
import UniformTypeIdentifiers

struct CoachView: View {
    @State private var controller: CoachController
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var showingImporter = false

    init(controller: CoachController = CoachController()) {
        _controller = State(initialValue: controller)
    }

    var body: some View {
        ZStack {
            BreathBackdrop().ignoresSafeArea()
            content
        }
        .navigationTitle("Coach")
        .navigationBarTitleDisplayMode(.inline)
        .task { controller.startDemoPlaybackIfNeeded() }
    }

    private var content: some View {
        // `currentTime` is published by the controller (playback polling task or
        // recorder drain) — the view only reads it, so reading it here makes the
        // whole screen re-render reactively. We must NOT mutate controller state
        // during this body, or it feeds back into an unbounded re-render loop.
        let time = controller.currentTime
        let state = controller.response.map {
            CoachingState.compute(at: time, from: $0)
        } ?? .empty
        let message = state.message(for: controller.phase,
                                    errorMessage: controller.errorMessage)

        return ScrollView {
            VStack(spacing: 20) {
                controlsRow
                if controller.phase != .recording && controller.phase != .analyzing {
                    uploadDemoRow
                }

                switch controller.phase {
                case .recording:
                    // Live wave — hero of recording. The wave IS the screen.
                    RecordingWaveformView(
                        levels: controller.recorder.recentLevels,
                        capacity: 720,
                        currentLevel: controller.recorder.currentLevel,
                        startedAt: controller.recordingStartedAt,
                        maxSeconds: controller.maxRecordingSeconds
                    )
                    CoachBannerView(message: message)
                case .idle:
                    // Inviting prompt; surface a banner only if something failed.
                    IdlePromptView()
                    if controller.errorMessage != nil {
                        CoachBannerView(message: message)
                    }
                case .analyzing:
                    AnalyzingView()
                case .ready, .playing, .ended:
                    // Analyzed wave is the hero — the wave IS the breathing
                    // pattern. Breath markers, regions, and a synced playhead.
                    timelineCard(time: time)
                    SideStatsView(state: state, phase: controller.phase)
                    CoachBannerView(message: message)
                }

                if controller.phase == .ended, let response = controller.response {
                    SessionSummaryView(response: response)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                if let response = controller.response {
                    AnalysisSectionView(response: response, currentTime: time)
                }
            }
            .padding(20)
            .animation(.spring(response: 0.4, dampingFraction: 0.8), value: controller.phase)
            .fileImporter(isPresented: $showingImporter,
                          allowedContentTypes: [.audio],
                          allowsMultipleSelection: false) { result in
                handleImport(result)
            }
        }
    }

    // MARK: - Shared pieces

    private func timelineCard(time: Double) -> some View {
        TimelineCardView(
            peaks: controller.waveformPeaks,
            events: controller.response?.predictedEvents ?? [],
            duration: controller.totalDuration,
            currentTime: time,
            isInteractive: controller.response != nil,
            onSeek: { controller.seek(to: $0) }
        )
    }

    // MARK: - Controls row

    private var controlsRow: some View {
        // Play and Record share remaining space equally; Reset hugs content.
        // Width is constrained by the parent VStack's .padding(20), so this
        // never bleeds past the screen edges.
        HStack(spacing: 8) {
            playButton.frame(maxWidth: .infinity)
            resetButton
            recordButton.frame(maxWidth: .infinity)
        }
    }

    // MARK: - Upload + demo clips (analyze without recording)

    private var uploadDemoRow: some View {
        HStack(spacing: 8) {
            Button { showingImporter = true } label: {
                pillLabel("Upload", system: "square.and.arrow.up")
            }
            Menu {
                ForEach(CoachController.demoClips) { clip in
                    Button(clip.title) {
                        if let url = clip.url { Task { await controller.analyze(fileURL: url) } }
                    }
                }
            } label: {
                pillLabel("Demos", system: "music.note.list")
            }
        }
    }

    private func pillLabel(_ title: String, system: String) -> some View {
        Label(title, systemImage: system)
            .font(.system(.callout, design: .rounded, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 14).padding(.vertical, 10)
            .frame(maxWidth: .infinity)
            .foregroundStyle(Color.breathInkPrimary)
            .background(Capsule().fill(.ultraThinMaterial))
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result, let picked = urls.first else { return }
        Task { await importAndAnalyze(picked) }
    }

    /// Copy the picked file out of its security-scoped location so it persists
    /// for playback, then analyze it on-device.
    private func importAndAnalyze(_ picked: URL) async {
        let scoped = picked.startAccessingSecurityScopedResource()
        defer { if scoped { picked.stopAccessingSecurityScopedResource() } }
        let ext = picked.pathExtension.isEmpty ? "m4a" : picked.pathExtension
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("upload_\(UUID().uuidString).\(ext)")
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.copyItem(at: picked, to: dest)
            await controller.analyze(fileURL: dest)
        } catch {
            // analyze() surfaces its own errors; copy failure is rare.
        }
    }

    private var playButton: some View {
        Button {
            if controller.phase == .playing { controller.pause() }
            else                            { controller.play() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: controller.phase == .playing ? "pause.fill" : "play.fill")
                Text(controller.phase == .playing ? "Pause" : "Play")
            }
            .font(.system(.callout, design: .rounded, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 14).padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .foregroundStyle(canPlay ? Color.white : Color.white.opacity(0.5))
            .background(
                Capsule().fill(LinearGradient.breathRibbon)
                    .opacity(canPlay ? 1 : 0.35)
            )
        }
        .disabled(!canPlay)
        .accessibilityLabel(controller.phase == .playing ? "Pause" : "Play")
    }

    private var resetButton: some View {
        Button { controller.reset() } label: {
            HStack(spacing: 6) {
                Image(systemName: "arrow.counterclockwise")
                Text("Reset")
            }
            .font(.system(.callout, design: .rounded, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 14).padding(.vertical, 12)
            .foregroundStyle(controller.response != nil
                             ? Color.breathInkPrimary
                             : Color.breathInkPrimary.opacity(0.35))
            .background(
                Capsule().fill(.ultraThinMaterial)
            )
        }
        .disabled(controller.response == nil)
        .accessibilityLabel("Reset")
    }

    private var recordButton: some View {
        let recording = controller.phase == .recording
        return Button {
            Task {
                if recording { await controller.stopAndAnalyze() }
                else         { await controller.startRecording() }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: recording ? "stop.fill" : "mic.fill")
                Text(recording ? "Stop" : "Record")
            }
            .font(.system(.callout, design: .rounded, weight: .semibold))
            .lineLimit(1)
            .padding(.horizontal, 14).padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .foregroundStyle(Color.white)
            .background(
                Capsule().fill(Color.breathMagenta.opacity(recording ? 1 : 0.85))
            )
            .overlay(recordingHalo(active: recording))
        }
        .disabled(controller.phase == .analyzing)
        .accessibilityLabel(recording ? "Stop recording" : "Start recording")
    }

    @ViewBuilder
    private func recordingHalo(active: Bool) -> some View {
        if reduceMotion {
            // No pulsing — a steady ring still signals "recording".
            Capsule()
                .strokeBorder(Color.breathMagenta.opacity(active ? 0.55 : 0), lineWidth: 2)
                .allowsHitTesting(false)
        } else {
            TimelineView(.animation(minimumInterval: 1.0/30.0, paused: !active)) { ctx in
                let t = ctx.date.timeIntervalSinceReferenceDate
                let phase = (sin(t * 2 * .pi / 1.4) + 1) / 2     // 0..1
                Capsule()
                    .strokeBorder(Color.breathMagenta.opacity(0.55), lineWidth: 2)
                    .opacity(active ? (0.4 + 0.4 * (1 - phase)) : 0)
            }
            .allowsHitTesting(false)
        }
    }

    private var canPlay: Bool {
        switch controller.phase {
        case .ready, .ended, .playing: return controller.response != nil
        default: return false
        }
    }
}

#Preview {
    NavigationStack { CoachView() }
}
