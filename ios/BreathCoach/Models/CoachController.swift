//
//  CoachController.swift
//  BreathCoach
//
//  Owns the audio + response lifecycle for Coach mode. Views read it via
//  @Bindable and rebuild from its current state — no view holds business logic.
//

import AVFoundation
import Foundation
import Observation
import SwiftUI

@MainActor
@Observable
final class CoachController {
    // MARK: - Public observable state

    private(set) var phase: CoachPhase = .idle
    private(set) var response: ProcessResponse?
    private(set) var errorMessage: String?

    /// Downsampled mono float samples for the timeline waveform.
    /// Lives here so we don't decode the WAV on every redraw.
    private(set) var waveformPeaks: [Float] = []

    /// Current playback position in seconds. The player is the source of truth
    /// while playing; `seek(to:)` writes it directly when scrubbing or paused.
    private(set) var currentTime: Double = 0

    /// Wall-clock time when the current recording began. Drives the recording
    /// view's elapsed-seconds counter without needing a separate timer.
    private(set) var recordingStartedAt: Date?

    /// DEBUG demo screens set this so the view starts playback from `.task` on
    /// appear rather than during construction. Stays false (no-op) in release.
    private var demoAutoplayPending = false

    // MARK: - Private

    let recorder = AudioRecorder()
    private var player: AVAudioPlayer?
    private var localRecordingURL: URL?
    private let waveformPeakCount = 720   // ~720 bars across the timeline

    /// Recordings auto-stop at this length so clips stay analyzable and a demo
    /// can't run away. The recording UI shows the remaining time.
    let maxRecordingSeconds: Double = 30
    // Internal task handles — not observed by the UI. `nonisolated(unsafe)` so
    // the nonisolated deinit can cancel them; `Task.cancel()` is thread-safe and
    // they're only assigned on the main actor.
    @ObservationIgnored private nonisolated(unsafe) var recordingCapTask: Task<Void, Never>?
    @ObservationIgnored private nonisolated(unsafe) var sessionObservers: [Task<Void, Never>] = []
    @ObservationIgnored private nonisolated(unsafe) var playbackTask: Task<Void, Never>?

    init() {
        observeAudioSession()
    }

    deinit {
        recordingCapTask?.cancel()
        sessionObservers.forEach { $0.cancel() }
        playbackTask?.cancel()
    }

    // MARK: - Recording lifecycle

    func startRecording() async {
        errorMessage = nil
        do {
            try await recorder.start()
            clearSession()          // drop the previous take now a new one is live
            phase = .recording
            recordingStartedAt = Date()
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            scheduleRecordingCap()
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    /// Auto-stop once a recording hits `maxRecordingSeconds`.
    private func scheduleRecordingCap() {
        recordingCapTask?.cancel()
        let limit = maxRecordingSeconds
        recordingCapTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(limit))
            guard !Task.isCancelled, let self, self.phase == .recording else { return }
            await self.stopAndAnalyze()
        }
    }

    /// Abandon the in-flight recording (e.g. an interruption) and surface why.
    private func abortRecording(reason: String) {
        recordingCapTask?.cancel()
        _ = recorder.stop()
        recordingStartedAt = nil
        errorMessage = reason
        phase = .idle
    }

    func stopAndAnalyze() async {
        recordingCapTask?.cancel()
        guard let url = recorder.stop() else {
            phase = .idle
            return
        }
        await analyze(fileURL: url)
    }

    /// Analyze any audio file on-device — shared by recording, file upload, and
    /// the bundled demo clips. Runs OnDeviceBreath (no server), then preps
    /// playback + the waveform from the same file.
    func analyze(fileURL url: URL) async {
        recordingCapTask?.cancel()
        recordingStartedAt = nil
        localRecordingURL = url
        errorMessage = nil
        phase = .analyzing
        do {
            let resp = try await OnDeviceBreath.shared.process(audioFileURL: url)
            response = resp
            waveformPeaks = (try? Self.loadPeaks(from: url, count: waveformPeakCount)) ?? []
            try preparePlayer(from: url)
            phase = .ready
            UINotificationFeedbackGenerator().notificationOccurred(.success)
        } catch {
            errorMessage = friendlyMessage(for: error)
            phase = .idle
        }
    }

    // MARK: - Demo clips (bundled audio, no recording needed)

    struct DemoClip: Identifiable, Hashable {
        let id: String          // resource base name
        let title: String       // shown in the menu
        var url: URL? { Bundle.main.url(forResource: id, withExtension: "wav") }
    }

    /// Bundled sung clips for demoing without a mic. Only those present in the
    /// app bundle are returned.
    static var demoClips: [DemoClip] {
        [
            DemoClip(id: "demo_clip",      title: "Sample — sung phrase"),
            DemoClip(id: "demo_soprano",   title: "Soprano — vibrato"),
            DemoClip(id: "demo_tenor",     title: "Tenor — straight"),
        ].filter { $0.url != nil }
    }

    // MARK: - Playback

    func play() {
        // Recorder.stop() deactivated the session; reactivate it for playback
        // before kicking the player. Without this, play() is silent and
        // currentTime never advances — which is why the playhead looked stuck.
        activateSessionForPlayback()
        guard let player else { return }
        if !player.isPlaying { player.play() }
        phase = .playing
        startPlaybackPolling()
    }

    /// Called from CoachView's `.task` on appear. Starts demo playback exactly
    /// once, on the controller instance the view actually displays.
    func startDemoPlaybackIfNeeded() {
        guard demoAutoplayPending else { return }
        demoAutoplayPending = false
        play()
    }

    /// Advance `currentTime` from the player's clock on a 60 Hz task — NOT from
    /// the view body. Reading `player.currentTime` during a SwiftUI update and
    /// writing it back to an `@Observable` property creates a re-render feedback
    /// loop (~1000 Hz) that blanks the screen. The view only *reads* currentTime.
    private func startPlaybackPolling() {
        playbackTask?.cancel()
        playbackTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self, self.phase == .playing, let player = self.player else { return }
                self.currentTime = player.currentTime
                if !player.isPlaying {
                    // Audio finished — flip to ended once and release audio focus.
                    self.phase = .ended
                    self.deactivatePlaybackSession()
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    return
                }
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
    }

    private func activateSessionForPlayback() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true, options: [])
        } catch {
            // Non-fatal: AVAudioPlayer can sometimes still play, just log.
            print("CoachController: playback session activation failed: \(error)")
        }
    }

    func pause() {
        playbackTask?.cancel()
        player?.pause()
        if phase == .playing { phase = .ready }
    }

    /// Move the playhead by tapping or scrubbing the timeline. Works whether
    /// playing or paused; the player follows so resuming continues from here.
    func seek(to time: Double) {
        guard totalDuration > 0 else { return }
        let clamped = min(max(0, time), totalDuration)
        player?.currentTime = clamped
        currentTime = clamped
    }

    func reset() {
        playbackTask?.cancel()
        player?.stop()
        player?.currentTime = 0
        currentTime = 0
        if response != nil { phase = .ready }
    }

    /// Drop the previous take's audio + analysis so a new recording starts clean.
    private func clearSession() {
        playbackTask?.cancel()
        player?.stop()
        player = nil
        response = nil
        waveformPeaks = []
        currentTime = 0
        localRecordingURL = nil
    }

    private func deactivatePlaybackSession() {
        try? AVAudioSession.sharedInstance()
            .setActive(false, options: [.notifyOthersOnDeactivation])
    }

    // MARK: - Audio session observation

    /// Degrade gracefully on interruptions (calls, alarms) and route changes
    /// (headphones unplugged) instead of recording silence or blasting audio.
    /// The `Task`s inherit this @MainActor context, so the non-Sendable
    /// `Notification` never leaves the main actor.
    private func observeAudioSession() {
        let interruptions = Task { [weak self] in
            for await note in NotificationCenter.default
                .notifications(named: AVAudioSession.interruptionNotification) {
                let raw = (note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt) ?? 0
                self?.handleInterruption(typeRaw: raw)
            }
        }
        let routes = Task { [weak self] in
            for await note in NotificationCenter.default
                .notifications(named: AVAudioSession.routeChangeNotification) {
                let raw = (note.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt) ?? 0
                self?.handleRouteChange(reasonRaw: raw)
            }
        }
        sessionObservers = [interruptions, routes]
    }

    private func handleInterruption(typeRaw: UInt) {
        guard let type = AVAudioSession.InterruptionType(rawValue: typeRaw) else { return }
        switch type {
        case .began:
            if phase == .recording { abortRecording(reason: "Recording interrupted — give it another go.") }
            else if phase == .playing { pause() }
        case .ended:
            break   // Don't auto-resume; let the singer choose to play again.
        @unknown default:
            break
        }
    }

    private func handleRouteChange(reasonRaw: UInt) {
        guard let reason = AVAudioSession.RouteChangeReason(rawValue: reasonRaw) else { return }
        // Headphones/Bluetooth removed mid-playback — pause, don't blast the speaker.
        if reason == .oldDeviceUnavailable, phase == .playing { pause() }
    }

    var totalDuration: Double {
        response?.durationSec ?? player?.duration ?? 0
    }

    // MARK: - Internals

    private func preparePlayer(from url: URL) throws {
        let player = try AVAudioPlayer(contentsOf: url)
        player.prepareToPlay()
        self.player = player
    }

    private func friendlyMessage(for error: Error) -> String {
        switch error {
        case BreathAPIError.serverUnreachable:
            return "Can't reach the model server. Is serve.py running, and is the address in Settings right?"
        case let apiError as BreathAPIError:
            return apiError.errorDescription ?? "Something went wrong. Try again."
        default:
            return (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    /// Downsample a WAV file into `count` peak values for waveform rendering.
    /// Each peak is max(|sample|) over its window — fast and visually meaningful.
    nonisolated static func loadPeaks(from url: URL, count: Int) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
        else { return [] }
        try file.read(into: buffer)
        guard let channelData = buffer.floatChannelData else { return [] }

        let samples = UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength))
        let total = samples.count
        guard total > 0 else { return [] }

        // Ceiling division so we never produce MORE than `count` buckets —
        // floor division overshoots and makes the timeline bars too thin.
        let bucket = max(1, (total + count - 1) / count)
        var peaks: [Float] = []
        peaks.reserveCapacity(count)
        var i = 0
        while i < total {
            let end = min(i + bucket, total)
            var maxAbs: Float = 0
            for j in i..<end {
                let v = abs(samples[j])
                if v > maxAbs { maxAbs = v }
            }
            peaks.append(maxAbs)
            i = end
        }
        return peaks
    }
}

#if DEBUG
// MARK: - Demo harness
//
// Populates a controller from the bundled `demo_clip` assets so the Coach
// screen can be rendered and screenshotted without a live server. Lives in
// this file so it can write the `private(set)` state. DEBUG only.

extension CoachController {
    enum DemoScenario {
        case playing(at: Double)   // frozen mid-playback at the given time
        case ended                 // playback finished; summary visible
    }

    static func demo(_ scenario: DemoScenario) -> CoachController {
        let controller = CoachController()
        controller.configureForDemo(scenario)
        return controller
    }

    /// The bundled demo clip decoded into a response — also used by demo screens
    /// that render a single component (e.g. the session summary).
    static func demoResponse() -> ProcessResponse? {
        guard let url = Bundle.main.url(forResource: "demo_clip", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ProcessResponse.self, from: data)
    }

    private func configureForDemo(_ scenario: DemoScenario) {
        guard let demo = Self.demoResponse() else { return }

        response = demo
        if let wav = Bundle.main.url(forResource: "demo_clip", withExtension: "wav") {
            waveformPeaks = (try? Self.loadPeaks(from: wav, count: waveformPeakCount)) ?? []
            // Real AVAudioPlayer — the demo uses the same play() path as record/upload.
            try? preparePlayer(from: wav)
        }

        switch scenario {
        case .playing(let time):
            currentTime = min(time, demo.durationSec)
            player?.currentTime = currentTime
            phase = .ready
            demoAutoplayPending = true   // CoachView's .task calls play() on appear
        case .ended:
            currentTime = demo.durationSec
            player?.currentTime = currentTime
            phase = .ended
        }
    }
}
#endif
