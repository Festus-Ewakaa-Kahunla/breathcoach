//
//  LogMel.swift
//  BreathCoach
//
//  On-device log-mel spectrogram — the EXACT preprocessing the CoreML model
//  expects. Mirrors the Python `compute_log_mel` (dataset.py) frame-for-frame:
//
//    frame  = audio[t*160 ..< t*160+400] * hann(400)
//    spec   = rfft(frame, n=512)            // 257 bins
//    power  = |spec|^2
//    mel    = filterbank(40x257) · power
//    logmel = log(mel + 1e-10)
//
//  The 40x257 filterbank and the 400-pt Hann window are loaded verbatim from
//  the bundled `preprocessing.json` (exported from Python) so the numbers match
//  bit-for-bit — we do NOT regenerate them here. Validate with parity_reference.json.
//

import Foundation
import Accelerate

// Immutable after init (all `let`), compute() mutates only locals — safe to share.
final class LogMel: @unchecked Sendable {
    static let shared = LogMel()

    // Params (mirror dataset.py). Loaded values override if present in JSON.
    private let sampleRate = 16_000
    private let nFFT = 512
    private let hop = 160
    private let win = 400
    private let nMels = 40
    private let nBins = 257           // rfft of 512 → 257
    private let logOffset: Float = 1e-10
    private let log2n: vDSP_Length = 9   // log2(512)

    private let hann: [Float]            // 400
    private let filterbank: [[Float]]    // 40 x 257
    private let fftSetup: FFTSetup

    /// vDSP_fft_zrip returns a result scaled by 2 vs. an unnormalized DFT
    /// (numpy's rfft). Divide by 2 to match. If the parity test shows a constant
    /// log offset across all bands, adjust THIS scale first.
    private let vdspScale: Float = 0.5

    private init() {
        guard let url = Bundle.main.url(forResource: "preprocessing", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let hannRaw = j["hann_window"] as? [Any],
              let fbRaw = j["mel_filterbank"] as? [[Any]] else {
            fatalError("LogMel: preprocessing.json missing or malformed in app bundle")
        }
        hann = hannRaw.map { ($0 as? NSNumber)?.floatValue ?? 0 }
        filterbank = fbRaw.map { row in row.map { ($0 as? NSNumber)?.floatValue ?? 0 } }
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("LogMel: failed to create FFT setup")
        }
        fftSetup = setup
    }

    deinit { vDSP_destroy_fftsetup(fftSetup) }

    /// 16 kHz mono samples → log-mel, returned as (flat row-major T*40, frameCount T).
    func compute(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let T = max(0, (samples.count - win) / hop + 1)
        guard T > 0 else { return ([], 0) }

        var out = [Float](repeating: 0, count: T * nMels)
        var realp = [Float](repeating: 0, count: nFFT / 2)   // 256
        var imagp = [Float](repeating: 0, count: nFFT / 2)
        var windowed = [Float](repeating: 0, count: nFFT)     // 512, zero-padded after 400

        for t in 0..<T {
            let start = t * hop
            for i in 0..<win { windowed[i] = samples[start + i] * hann[i] }
            for i in win..<nFFT { windowed[i] = 0 }

            realp.withUnsafeMutableBufferPointer { rp in
                imagp.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    windowed.withUnsafeBufferPointer { wp in
                        wp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { cp in
                            vDSP_ctoz(cp, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                    // Power spectrum, 257 bins. zrip packing:
                    //   rp[0] = DC (real),  ip[0] = Nyquist (real),  (rp[k],ip[k]) = bin k for 1..255
                    var power = [Float](repeating: 0, count: nBins)
                    let s = vdspScale
                    power[0] = (rp[0] * s) * (rp[0] * s)
                    power[nBins - 1] = (ip[0] * s) * (ip[0] * s)
                    for k in 1..<(nFFT / 2) {
                        let re = rp[k] * s, im = ip[k] * s
                        power[k] = re * re + im * im
                    }

                    // mel = filterbank · power  → log
                    for m in 0..<nMels {
                        var acc: Float = 0
                        filterbank[m].withUnsafeBufferPointer { fb in
                            power.withUnsafeBufferPointer { pw in
                                vDSP_dotpr(fb.baseAddress!, 1, pw.baseAddress!, 1, &acc, vDSP_Length(nBins))
                            }
                        }
                        out[t * nMels + m] = log(acc + logOffset)
                    }
                }
            }
        }
        return (out, T)
    }
}
