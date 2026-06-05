//
//  CoachingState.swift
//  BreathCoach
//
//  Per-frame derivation of "how is this singer doing right now?" from the
//  server's ProcessResponse + current audio time. Mirrors the web app's
//  coachingState() function (see ui.md §6, index.html lines 431–437).
//

import Foundation
import SwiftUI

enum CoachPhase: Equatable {
    case idle           // no audio loaded yet
    case recording      // mic open, time accumulating
    case analyzing      // POST /process in flight
    case ready          // response in hand, not yet playing
    case playing        // AVAudioPlayer playing, ring + banner animating
    case ended          // playback finished, summary visible
}

enum CoachingTone {
    case neutral        // no special color — default glass
    case breath         // currently inside a detected breath
    case good           // healthy phrase length
    case warn           // short or amber zone
    case danger         // 30s+ without a breath, or an error
}

struct CoachingState: Equatable {
    var curPhraseDur: Double = 0
    var lastPhraseDur: Double? = nil
    var breathCount: Int = 0
    var inBreath: Bool = false

    static let empty = CoachingState()

    /// Compute coaching state at a given playback time.
    /// O(N) over events + phrases — fine for the lengths we deal with (<200 frames of events).
    static func compute(at time: Double, from response: ProcessResponse) -> CoachingState {
        // The server tells us which event source it derived phrases from.
        // For breath counting we use that same source so the displayed count
        // matches the phrase boundaries.
        let useRuinskiy = (response.phrasesFrom == "ruinskiy")
        let starts: [Double] = useRuinskiy
            ? response.ruinskiyEvents.map(\.startSec)
            : response.predictedEvents.map(\.startSec)
        let ends:   [Double] = useRuinskiy
            ? response.ruinskiyEvents.map(\.endSec)
            : response.predictedEvents.map(\.endSec)

        var breathCount = 0
        var inBreath = false
        for i in starts.indices {
            if ends[i] <= time { breathCount += 1 }
            if starts[i] <= time && time < ends[i] { inBreath = true }
        }

        var curPhraseDur: Double = 0
        var lastPhraseDur: Double? = nil
        for (i, ph) in response.phraseEvents.enumerated() {
            if ph.startSec <= time && time < ph.endSec {
                curPhraseDur = time - ph.startSec
                if i > 0 { lastPhraseDur = response.phraseEvents[i-1].durationSec }
                break
            } else if time >= ph.endSec {
                lastPhraseDur = ph.durationSec
            }
        }

        return CoachingState(curPhraseDur: curPhraseDur,
                             lastPhraseDur: lastPhraseDur,
                             breathCount: breathCount,
                             inBreath: inBreath)
    }
}

// MARK: - Banner copy (verbatim from web app — see ui.md §4.3)

struct CoachingMessage {
    let text: String
    let tone: CoachingTone
}

extension CoachingState {
    /// The coaching banner text + tone for the current state. Copy mirrors
    /// `phraseQuality()` in `web/index.html` so the iOS app feels like the web.
    func message(for phase: CoachPhase, errorMessage: String? = nil) -> CoachingMessage {
        if let msg = errorMessage {
            return .init(text: msg, tone: .danger)
        }
        switch phase {
        case .idle:      return .init(text: "Press record to begin.", tone: .neutral)
        case .recording: return .init(text: "Listening…", tone: .neutral)
        case .analyzing: return .init(text: "Reading your phrasing…", tone: .neutral)
        case .ready:     return .init(text: "Press play to hear your coaching.", tone: .neutral)
        case .ended:     return .init(text: "Session complete — see your summary below.", tone: .good)
        case .playing:
            if inBreath {
                return .init(text: "🫁  Breath — phrase reset.", tone: .breath)
            }
            switch curPhraseDur {
            case ..<1.2:  return .init(text: "Carrying the line…", tone: .neutral)
            case ..<2:    return .init(text: "Short phrase — try to carry the line a little longer.", tone: .warn)
            case ..<4:    return .init(text: "Nice — see if you can stretch the next one further.", tone: .warn)
            case ..<8:    return .init(text: "Comfortable, well-supported phrasing.", tone: .good)
            case ..<12:   return .init(text: "Strong breath support — lovely long phrase.", tone: .good)
            case ..<30:   return .init(text: "Impressive control on that long phrase.", tone: .breath)
            default:      return .init(text: "30+ seconds without a breath — ease off before you strain.", tone: .danger)
            }
        }
    }

    /// The ring arc's color gradient picks itself by current coaching state.
    /// See ui.md §4.2.
    func ringGradient() -> [Color] {
        if inBreath {
            return [.breathTeal, .breathIndigo]
        }
        switch curPhraseDur {
        case ..<2:    return [.breathAmber, .breathRed.opacity(0.85)]
        case 12..<30: return [.breathIndigo, .breathMagenta]
        case 30...:   return [.breathRed, .breathMagenta]
        default:      return [.breathTeal, .breathIndigo]
        }
    }
}

