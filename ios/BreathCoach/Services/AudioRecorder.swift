//
//  AudioRecorder.swift
//  BreathCoach
//
//  Records via AVAudioEngine so we can tap the input node for live peaks AND
//  write the same buffers to a WAV file.
//
//  The audio thread runs on a non-main, non-actor context. To make that safe
//  with Swift 6 strict concurrency, all per-buffer work happens inside a
//  separate `RecordingTap` class (@unchecked Sendable, lock-protected) that
//  doesn't touch any actor-isolated state. The AudioRecorder polls that
//  RecordingTap from main on a Timer and republishes into @Observable state.
//

import AVFoundation
import Foundation
import Observation

// MARK: - Lock-protected audio-thread state

/// Owns the AVAudioFile + peak ring buffer. Lives off the main actor so the
/// tap closure can call into it freely. Reading happens via `snapshot()` from
/// the main actor — fully thread-safe via the lock.
private final class RecordingTap: @unchecked Sendable {
    private let lock = NSLock()
    private let file: AVAudioFile
    private let capacity: Int
    private let peaksPerBuffer: Int

    // Mutable state — only touch under `lock`.
    private var peaks: [Float] = []
    private var lastPeak: Float = 0

    init(file: AVAudioFile, capacity: Int, peaksPerBuffer: Int) {
        self.file = file
        self.capacity = capacity
        self.peaksPerBuffer = peaksPerBuffer
        self.peaks.reserveCapacity(capacity)
    }

    /// Called on the real-time audio thread, once per buffer.
    func process(_ buffer: AVAudioPCMBuffer) {
        // AVAudioFile.write is documented as safe to call from a single
        // thread; the tap callback is consistently on the same dispatch
        // queue per source, so this is fine.
        try? file.write(from: buffer)

        let newPeaks = Self.computePeaks(from: buffer, count: peaksPerBuffer)
        guard !newPeaks.isEmpty else { return }

        lock.lock()
        peaks.append(contentsOf: newPeaks)
        if peaks.count > capacity {
            peaks.removeFirst(peaks.count - capacity)
        }
        lastPeak = newPeaks.last ?? lastPeak
        lock.unlock()
    }

    /// Called on main when the recorder's drain timer fires.
    func snapshot() -> (peaks: [Float], last: Float) {
        lock.lock()
        let copy = peaks
        let last = lastPeak
        lock.unlock()
        return (copy, last)
    }

    /// Closes the underlying file. Idempotent — safe to call from any thread.
    func close() {
        // AVAudioFile closes on deinit; we just drop our reference by
        // letting the wrapping recorder release us.
    }

    // MARK: - Peak math (static so it's trivially Sendable)

    private static func computePeaks(from buffer: AVAudioPCMBuffer,
                                     count: Int) -> [Float] {
        guard let channelData = buffer.floatChannelData?[0] else { return [] }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return [] }
        let chunkSize = max(1, frames / count)
        var out: [Float] = []
        out.reserveCapacity(count)
        for c in 0..<count {
            let start = c * chunkSize
            let end = min(start + chunkSize, frames)
            var peak: Float = 0
            for i in start..<end {
                let v = abs(channelData[i])
                if v > peak { peak = v }
            }
            // Mild gain so quiet humming is still visible. Capped at 1.
            out.append(min(1, peak * 1.6))
        }
        return out
    }
}

// MARK: - Main-actor recorder

@MainActor
@Observable
final class AudioRecorder {
    enum RecorderError: LocalizedError {
        case permissionDenied
        case sessionConfigurationFailed(underlying: Error)
        case engineSetupFailed(underlying: Error)

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Microphone permission is off. Enable it in Settings → BreathCoach."
            case .sessionConfigurationFailed(let underlying):
                return "Couldn't set up audio session: \(underlying.localizedDescription)"
            case .engineSetupFailed(let underlying):
                return "Couldn't start the recorder: \(underlying.localizedDescription)"
            }
        }
    }

    // MARK: - Observable state

    private(set) var isRecording = false
    private(set) var lastRecordingURL: URL?
    private(set) var recentLevels: [Float] = []
    private(set) var currentLevel: Float = 0

    // MARK: - Tunables

    /// ~1.9 s of visible audio at ~376 peaks/sec from 48 kHz / 1024-frame
    /// buffers × 8 peaks per buffer.
    private let capacity = 720
    private let peaksPerBuffer = 8

    /// Polling rate at which we drain the tap into @Observable state.
    /// 30 Hz is enough to look live without spamming SwiftUI invalidations.
    private let drainHz: Double = 30

    // MARK: - Private

    private var engine: AVAudioEngine?
    private var tap: RecordingTap?
    private var drainTimer: Timer?

    // MARK: - Permission

    func ensurePermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: return true
        case .denied:  return false
        case .undetermined:
            return await AVAudioApplication.requestRecordPermission()
        @unknown default:
            return false
        }
    }

    // MARK: - Recording lifecycle

    func start() async throws {
        guard await ensurePermission() else { throw RecorderError.permissionDenied }
        try configureSession()

        do {
            let engine = AVAudioEngine()
            let inputNode = engine.inputNode
            let inputFormat = inputNode.outputFormat(forBus: 0)
            guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
                throw NSError(
                    domain: "AudioRecorder", code: -2,
                    userInfo: [NSLocalizedDescriptionKey:
                                "Input format unavailable. Try again — the audio session may not be ready."]
                )
            }

            let url = Self.makeRecordingURL()
            // Write WAV at the input's sample rate. The server resamples to
            // 16 kHz internally, so we skip the on-device conversion.
            // 16-bit int PCM keeps file size manageable.
            let fileSettings: [String: Any] = [
                AVFormatIDKey:             kAudioFormatLinearPCM,
                AVSampleRateKey:           inputFormat.sampleRate,
                AVNumberOfChannelsKey:     inputFormat.channelCount,
                AVLinearPCMBitDepthKey:    16,
                AVLinearPCMIsFloatKey:     false,
                AVLinearPCMIsBigEndianKey: false,
            ]
            let file = try AVAudioFile(forWriting: url, settings: fileSettings)

            let tap = RecordingTap(file: file,
                                   capacity: capacity,
                                   peaksPerBuffer: peaksPerBuffer)

            // Reset visualization state up-front so the first drain doesn't
            // read leftover peaks from a previous take.
            recentLevels.removeAll(keepingCapacity: true)
            currentLevel = 0

            // Build the tap block via a `nonisolated` helper so Swift doesn't
            // inherit @MainActor isolation from start()'s context. Without
            // this, the closure runs on the audio thread but Swift inserts a
            // runtime check expecting main, which fires _swift_task_checkIsolatedSwift
            // → SIGTRAP on the first audio buffer.
            inputNode.installTap(onBus: 0,
                                 bufferSize: 1024,
                                 format: inputFormat,
                                 block: Self.makeTapBlock(for: tap))

            try engine.start()
            startDrain()

            self.engine = engine
            self.tap = tap
            self.lastRecordingURL = url
            self.isRecording = true
        } catch let error as RecorderError {
            throw error
        } catch {
            throw RecorderError.engineSetupFailed(underlying: error)
        }
    }

    @discardableResult
    func stop() -> URL? {
        stopDrain()
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine = nil
        tap = nil                       // releases AVAudioFile → WAV header flushes
        isRecording = false
        currentLevel = 0
        try? AVAudioSession.sharedInstance()
            .setActive(false, options: [.notifyOthersOnDeactivation])
        return lastRecordingURL
    }

    // MARK: - Drain

    private func startDrain() {
        drainTimer?.invalidate()
        let interval = 1.0 / drainHz
        drainTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            // Timer fires on the runloop the AudioRecorder was started on,
            // which is the main runloop (start() is @MainActor). Safe to
            // touch @Observable state directly.
            Task { @MainActor [weak self] in
                self?.drainTick()
            }
        }
    }

    private func stopDrain() {
        drainTimer?.invalidate()
        drainTimer = nil
    }

    private func drainTick() {
        guard let tap else { return }
        let (peaks, last) = tap.snapshot()
        recentLevels = peaks
        currentLevel = last
    }

    // MARK: - Tap block factory

    /// Build the audio-thread tap closure from a non-isolated context. The
    /// `nonisolated` keyword strips the implicit @MainActor inheritance that
    /// closures built inside `start()` would otherwise carry.
    nonisolated private static func makeTapBlock(
        for tap: RecordingTap
    ) -> @Sendable (AVAudioPCMBuffer, AVAudioTime) -> Void {
        return { buffer, _ in
            tap.process(buffer)
        }
    }

    // MARK: - Session + URL

    private func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord,
                                    mode: .measurement,
                                    options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true, options: [])
        } catch {
            throw RecorderError.sessionConfigurationFailed(underlying: error)
        }
    }

    private static func makeRecordingURL() -> URL {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("breathcoach-\(stamp).wav")
    }
}
