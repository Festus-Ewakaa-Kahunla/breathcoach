//
//  BreathAPI.swift
//  BreathCoach
//
//  Thin client for the FastAPI server in ../breathcoach.
//  Replace `baseURL` with your Mac's local IP before running on device.
//

import Foundation
import Observation

@MainActor
@Observable
final class ServerHealth {
    enum Status { case unknown, checking, reachable, unreachable }

    private(set) var status: Status = .unknown
    private(set) var modelVersion: String? = nil
    private(set) var lastError: String? = nil

    var isReachable: Bool { status == .reachable }

    func check() async {
        status = .checking
        do {
            let info = try await BreathAPI.shared.health()
            modelVersion = info.modelVersion
            status = info.ok ? .reachable : .unreachable
            lastError = info.ok ? nil : "Server responded but isn't ready."
        } catch {
            status = .unreachable
            modelVersion = nil
            lastError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }
}

enum BreathAPIError: LocalizedError {
    case badResponse
    case decodingFailed
    case serverUnreachable

    var errorDescription: String? {
        switch self {
        case .badResponse:       return "The server replied, but something was off."
        case .decodingFailed:    return "Couldn't read the server's reply."
        case .serverUnreachable: return "Can't reach the model server."
        }
    }
}

struct HealthResponse: Decodable {
    let ok: Bool
    let modelVersion: String?
    let uptimeSec: Double?

    enum CodingKeys: String, CodingKey {
        case ok
        case modelVersion = "model_version"
        case uptimeSec = "uptime_sec"
    }
}

struct BreathEvent: Decodable, Identifiable {
    var id: String { "\(startSec)-\(endSec)" }
    let startSec: Double
    let endSec: Double
    let confidence: Double?

    enum CodingKeys: String, CodingKey {
        case startSec = "start_sec"
        case endSec = "end_sec"
        case confidence
    }
}

struct DetectResponse: Decodable {
    let durationSec: Double
    let events: [BreathEvent]
    let modelVersion: String

    enum CodingKeys: String, CodingKey {
        case durationSec = "duration_sec"
        case events
        case modelVersion = "model_version"
    }
}

// MARK: - /process response (Coach mode)

/// A single phrase between breaths.
struct PhraseEvent: Decodable, Identifiable {
    var id: String { "\(startSec)-\(endSec)" }
    let startSec: Double
    let endSec: Double
    let durationSec: Double

    enum CodingKeys: String, CodingKey {
        case startSec    = "start_sec"
        case endSec      = "end_sec"
        case durationSec = "duration_sec"
    }
}

/// Rusinkiy/Lavner DSP baseline event (has a confidence score). Used in
/// Analysis comparisons vs. the neural model.
struct BaselineEvent: Decodable, Identifiable {
    var id: String { "\(startSec)-\(endSec)" }
    let startSec: Double
    let endSec: Double
    let score: Double?

    enum CodingKeys: String, CodingKey {
        case startSec = "start_sec"
        case endSec   = "end_sec"
        case score
    }
}

struct ModelMeta: Decodable {
    let hidden: Int?
    let params: Int?
    let inferenceMs: Double?
    let perFrameMs: Double?

    enum CodingKeys: String, CodingKey {
        case hidden, params
        case inferenceMs = "inference_ms"
        case perFrameMs  = "per_frame_ms"
    }
}

struct ProcessResponse: Decodable {
    let audioFile: String?
    let spectrogramFile: String?
    let durationSec: Double
    let sampleRate: Int
    let hopSec: Double
    let nFrames: Int
    let breathProb: [Double]
    let voicedProb: [Double]
    let pitchNorm: [Double]
    let predictedEvents: [BreathEvent]
    let ruinskiyEvents: [BaselineEvent]
    let phraseEvents: [PhraseEvent]
    let threshold: Double
    let phrasesFrom: String
    let modelMeta: ModelMeta

    enum CodingKeys: String, CodingKey {
        case audioFile        = "audio_file"
        case spectrogramFile  = "spectrogram_file"
        case durationSec      = "duration_sec"
        case sampleRate       = "sample_rate"
        case hopSec           = "hop_sec"
        case nFrames          = "n_frames"
        case breathProb       = "breath_prob"
        case voicedProb       = "voiced_prob"
        case pitchNorm        = "pitch_norm"
        case predictedEvents  = "predicted_events"
        case ruinskiyEvents   = "ruinskiy_events"
        case phraseEvents     = "phrase_events"
        case threshold
        case phrasesFrom      = "phrases_from"
        case modelMeta        = "model_meta"
    }
}

final class BreathAPI: Sendable {
    static let shared = BreathAPI()

    /// Configured by the user in Settings (persisted via `ServerStore`), so the
    /// app survives Wi-Fi changes without a rebuild.
    static var baseURL: URL { ServerStore.resolvedURL() }

    private static let healthTimeout: TimeInterval = 6
    private static let processTimeout: TimeInterval = 30

    func health() async throws -> HealthResponse {
        let url = Self.baseURL.appendingPathComponent("health")
        var request = URLRequest(url: url)
        request.timeoutInterval = Self.healthTimeout
        let (data, response) = try await send(request, mapErrorsTo: .serverUnreachable)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BreathAPIError.serverUnreachable
        }
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }

    func detect(audioFileURL: URL) async throws -> DetectResponse {
        let url = Self.baseURL.appendingPathComponent("detect")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = try multipartBody(boundary: boundary, fileURL: audioFileURL, fieldName: "audio")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BreathAPIError.badResponse
        }
        return try JSONDecoder().decode(DetectResponse.self, from: data)
    }

    /// Full Coach pipeline: raw WAV → predictions, events, spectrogram path, meta.
    /// The server's `/process` route reads the request body as raw audio bytes
    /// (no multipart) because it predates the iOS client. We honor that here.
    func process(audioFileURL: URL) async throws -> ProcessResponse {
        let url = Self.baseURL.appendingPathComponent("process")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.processTimeout
        request.setValue("audio/wav", forHTTPHeaderField: "Content-Type")
        request.httpBody = try Data(contentsOf: audioFileURL)

        let (data, response) = try await send(request, mapErrorsTo: .serverUnreachable)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw BreathAPIError.badResponse
        }
        do {
            let result = try JSONDecoder().decode(ProcessResponse.self, from: data)
            // A clip with no duration can't be played back or drawn — treat it
            // as a bad response rather than rendering a broken timeline.
            guard result.durationSec > 0 else { throw BreathAPIError.badResponse }
            return result
        } catch let error as BreathAPIError {
            throw error
        } catch {
            throw BreathAPIError.decodingFailed
        }
    }

    /// Run a request, translating connectivity-level `URLError`s into a single
    /// app error so callers can show one friendly message.
    private func send(_ request: URLRequest,
                      mapErrorsTo fallback: BreathAPIError) async throws -> (Data, URLResponse) {
        do {
            return try await URLSession.shared.data(for: request)
        } catch is URLError {
            throw fallback
        }
    }

    /// Compose a URL for a file referenced by `audio_file` / `spectrogram_file`.
    /// On-device, these are local `file://` URLs (or absolute paths) — pass them
    /// through. Server paths are resolved relative to `recordings/`.
    func staticFileURL(forRelativePath relativePath: String) -> URL {
        if relativePath.hasPrefix("file:"), let u = URL(string: relativePath) { return u }
        if relativePath.hasPrefix("/") { return URL(fileURLWithPath: relativePath) }
        return Self.baseURL
            .appendingPathComponent("recordings")
            .appendingPathComponent((relativePath as NSString).lastPathComponent)
    }

    private func multipartBody(boundary: String, fileURL: URL, fieldName: String) throws -> Data {
        var body = Data()
        let filename = fileURL.lastPathComponent
        let mime = "audio/wav"
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(mime)\r\n\r\n".data(using: .utf8)!)
        body.append(try Data(contentsOf: fileURL))
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }
}
