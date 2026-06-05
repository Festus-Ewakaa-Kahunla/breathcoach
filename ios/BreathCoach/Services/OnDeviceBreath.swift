//
//  OnDeviceBreath.swift
//  BreathCoach
//
//  Fully on-device breath analysis — no server. Drop-in replacement for
//  `BreathAPI.shared.process(audioFileURL:)`: same signature, same
//  `ProcessResponse` output, so Coach is unchanged except which object it calls.
//
//  Pipeline:  audio file → 16 kHz mono → LogMel (1,T,40) → CoreML → breath/voiced
//             → peak-pick → breath events → phrases → ProcessResponse
//
//  Requires in the app bundle: BreathCoach.mlpackage (Xcode compiles → .mlmodelc)
//  and preprocessing.json (loaded by LogMel). Validate with parity_reference.json.
//

import Foundation
import AVFoundation
import CoreML
import UIKit
import CoreGraphics

enum OnDeviceError: LocalizedError {
    case modelMissing, decodeFailed, emptyAudio
    var errorDescription: String? {
        switch self {
        case .modelMissing: return "On-device model not found in the app bundle."
        case .decodeFailed: return "Couldn't read the audio file."
        case .emptyAudio:   return "The recording was empty."
        }
    }
}

// Immutable after init; MLModel.prediction is thread-safe — safe to share.
final class OnDeviceBreath: @unchecked Sendable {
    static let shared = OnDeviceBreath()

    private let model: MLModel?

    private init() {
        // Xcode compiles BreathCoach.mlpackage → BreathCoach.mlmodelc in the bundle.
        if let url = Bundle.main.url(forResource: "BreathCoach", withExtension: "mlmodelc") {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
            model = try? MLModel(contentsOf: url, configuration: cfg)
        } else {
            model = nil
        }
    }

    /// Same shape as BreathAPI.process — swap one for the other in CoachController.
    func process(audioFileURL: URL) async throws -> ProcessResponse {
        guard let model else { throw OnDeviceError.modelMissing }

        let samples = try Self.load16kMono(audioFileURL)
        guard samples.count > 400 else { throw OnDeviceError.emptyAudio }
        let durationSec = Double(samples.count) / 16_000.0

        // ── preprocessing ──
        let (melFlat, T) = LogMel.shared.compute(samples)
        guard T > 0 else { throw OnDeviceError.emptyAudio }

        // Render the log-mel to a PNG on-device so the Analysis panel shows a
        // real spectrogram (no server). Returns a file:// path or nil.
        let specPath = Self.renderSpectrogramPNG(mel: melFlat, frames: T, nMels: 40)

        // ── pack into MLMultiArray (1, T, 40) ──
        let mel = try MLMultiArray(shape: [1, NSNumber(value: T), 40], dataType: .float32)
        let mptr = mel.dataPointer.bindMemory(to: Float.self, capacity: T * 40)
        for i in 0..<(T * 40) { mptr[i] = melFlat[i] }

        // ── inference ──
        let t0 = Date()
        let provider = try MLDictionaryFeatureProvider(dictionary: ["mel": MLFeatureValue(multiArray: mel)])
        let out = try await model.prediction(from: provider)
        let inferenceMs = Date().timeIntervalSince(t0) * 1000.0
        guard let breathArr = out.featureValue(for: "breath")?.multiArrayValue,
              let voicedArr = out.featureValue(for: "voiced")?.multiArrayValue else {
            throw OnDeviceError.decodeFailed
        }
        let breath = Self.toDoubles(breathArr)
        let voiced = Self.toDoubles(voicedArr)

        // ── events → phrases ──
        let events = Self.peakPick(breath)
        let phr = Self.phrases(events, duration: durationSec)

        let breathEvents = events.map { BreathEvent(startSec: $0.0, endSec: $0.1, confidence: nil) }
        let phraseEvents = phr.map { PhraseEvent(startSec: $0.0, endSec: $0.1, durationSec: $0.2) }
        let meta = ModelMeta(hidden: 8, params: 15_705,
                             inferenceMs: inferenceMs,
                             perFrameMs: inferenceMs / Double(max(1, T)))

        return ProcessResponse(
            audioFile: nil, spectrogramFile: specPath,
            durationSec: durationSec, sampleRate: 16_000, hopSec: 0.01, nFrames: T,
            breathProb: breath, voicedProb: voiced, pitchNorm: [],
            predictedEvents: breathEvents, ruinskiyEvents: [], phraseEvents: phraseEvents,
            threshold: 0.25, phrasesFrom: "breath_head", modelMeta: meta)
    }

    // MARK: - Audio → 16 kHz mono Float

    private static func load16kMono(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                            sampleRate: 16_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat,
                                           frameCapacity: AVAudioFrameCount(file.length)) else {
            throw OnDeviceError.decodeFailed
        }
        try file.read(into: inBuf)

        let ratio = 16_000.0 / inFormat.sampleRate
        let outCap = AVAudioFrameCount(Double(file.length) * ratio) + 2048
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCap) else {
            throw OnDeviceError.decodeFailed
        }
        var consumed = false
        var convErr: NSError?
        converter.convert(to: outBuf, error: &convErr) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true; status.pointee = .haveData; return inBuf
        }
        if let convErr { throw convErr }
        let n = Int(outBuf.frameLength)
        guard n > 0, let ch = outBuf.floatChannelData else { throw OnDeviceError.emptyAudio }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
    }

    /// Read an MLMultiArray's scalars as Doubles, honoring its element type.
    /// The model is exported FP16, so its outputs are `.float16` — binding the
    /// raw buffer to `Float` (FP32) misreads two halves as one float and runs off
    /// the end of the buffer, yielding ±1e38 / NaN garbage and zero detections.
    private static func toDoubles(_ arr: MLMultiArray) -> [Double] {
        let n = arr.count
        switch arr.dataType {
        case .float32:
            let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: n)
            return (0..<n).map { Double(ptr[$0]) }
        case .double:
            let ptr = arr.dataPointer.bindMemory(to: Double.self, capacity: n)
            return (0..<n).map { ptr[$0] }
        default:
            // .float16 (and any other type) — the boxed subscript converts safely.
            return (0..<n).map { arr[$0].doubleValue }
        }
    }

    // MARK: - Peak-pick (port of precompute_predictions.peak_events defaults)

    private static func peakPick(_ probs: [Double], hopSec: Double = 0.01) -> [(Double, Double)] {
        let n = probs.count
        guard n > 2 else { return [] }
        let w = max(1, Int((0.080 / hopSec).rounded()))     // ~80 ms smooth
        var smooth = probs
        if w > 1 {
            smooth = (0..<n).map { i in
                let lo = max(0, i - w / 2), hi = min(n, i + w / 2 + 1)
                var s = 0.0; for k in lo..<hi { s += probs[k] }
                return s / Double(hi - lo)
            }
        }
        let baseline = median(smooth)
        let minProm = 0.12
        let minDist = max(1, Int((1.0 / hopSec).rounded()))  // 1 s → 100 frames

        var peaks: [Int] = []
        for i in 1..<(n - 1) where smooth[i] > smooth[i - 1] && smooth[i] >= smooth[i + 1] {
            guard smooth[i] - baseline >= minProm else { continue }
            if let last = peaks.last, i - last < minDist {
                if smooth[i] > smooth[last] { peaks[peaks.count - 1] = i }   // keep the taller
            } else {
                peaks.append(i)
            }
        }

        let relHeight = 0.6
        var events: [(Double, Double)] = []
        for p in peaks {
            let cutoff = baseline + (smooth[p] - baseline) * relHeight
            var l = p; while l > 0 && smooth[l] > cutoff { l -= 1 }
            var r = p; while r < n - 1 && smooth[r] > cutoff { r += 1 }
            events.append((Double(l) * hopSec, Double(r) * hopSec))
        }
        return events
    }

    // MARK: - Spectrogram render (mel → PNG file, brand colormap)

    /// Renders the (T×nMels) log-mel to a PNG in a temp dir. Returns a file:// URL
    /// string (consumed by AnalysisSectionView via staticFileURL passthrough), or nil.
    private static func renderSpectrogramPNG(mel: [Float], frames T: Int, nMels: Int) -> String? {
        guard T > 0, mel.count >= T * nMels else { return nil }

        // Downsample columns so the image isn't absurdly wide on long clips.
        let maxW = 1024
        let stride = max(1, T / maxW)
        let W = (T + stride - 1) / stride
        let H = nMels

        // Normalize log-mel to 0..1 over the clip.
        var lo: Float = .greatestFiniteMagnitude, hi: Float = -.greatestFiniteMagnitude
        for v in mel { if v < lo { lo = v }; if v > hi { hi = v } }
        let range = max(1e-6, hi - lo)

        var pixels = [UInt8](repeating: 0, count: W * H * 4)   // RGBA8
        for x in 0..<W {
            let t = min(T - 1, x * stride)
            for band in 0..<H {
                let v = (mel[t * nMels + band] - lo) / range     // 0..1
                let (r, g, b) = colormap(v)
                // flip vertically: low freq (band 0) at the bottom
                let y = H - 1 - band
                let i = (y * W + x) * 4
                pixels[i] = r; pixels[i + 1] = g; pixels[i + 2] = b; pixels[i + 3] = 255
            }
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &pixels, width: W, height: H, bitsPerComponent: 8,
                                  bytesPerRow: W * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cg = ctx.makeImage() else { return nil }
        let img = UIImage(cgImage: cg)
        guard let png = img.pngData() else { return nil }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("spectrogram_\(UUID().uuidString).png")
        do { try png.write(to: url); return url.absoluteString } catch { return nil }
    }

    /// Dark → teal → indigo → magenta ramp (matches the BreathCoach palette).
    private static func colormap(_ v: Float) -> (UInt8, UInt8, UInt8) {
        let stops: [(Float, (Float, Float, Float))] = [
            (0.0, (11, 11, 20)),      // #0B0B14
            (0.4, (28, 114, 147)),    // #1C7293
            (0.7, (108, 123, 255)),   // #6C7BFF
            (1.0, (255, 94, 160)),    // #FF5EA0
        ]
        let x = min(1, max(0, v))
        for i in 0..<(stops.count - 1) {
            let (a, ca) = stops[i], (b, cb) = stops[i + 1]
            if x <= b {
                let f = (x - a) / max(1e-6, b - a)
                return (UInt8(ca.0 + (cb.0 - ca.0) * f),
                        UInt8(ca.1 + (cb.1 - ca.1) * f),
                        UInt8(ca.2 + (cb.2 - ca.2) * f))
            }
        }
        let last = stops.last!.1
        return (UInt8(last.0), UInt8(last.1), UInt8(last.2))
    }

    private static func median(_ x: [Double]) -> Double {
        guard !x.isEmpty else { return 0 }
        let s = x.sorted(); let m = s.count / 2
        return s.count % 2 == 0 ? (s[m - 1] + s[m]) / 2 : s[m]
    }

    // MARK: - Phrases (sung spans between breaths)

    private static func phrases(_ events: [(Double, Double)], duration: Double) -> [(Double, Double, Double)] {
        var out: [(Double, Double, Double)] = []
        var cursor = 0.0
        for ev in events.sorted(by: { $0.0 < $1.0 }) {
            if ev.0 > cursor { out.append((cursor, ev.0, ev.0 - cursor)) }
            cursor = max(cursor, ev.1)
        }
        if duration > cursor { out.append((cursor, duration, duration - cursor)) }
        return out
    }
}
