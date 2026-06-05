//
//  BreathCoachApp.swift
//  BreathCoach
//
//  iOS 26 Liquid Glass — entry point.
//

import SwiftUI

@main
struct BreathCoachApp: App {
    @State private var serverHealth = ServerHealth()
    #if DEBUG
    // Built once. rootView is recomputed whenever the App body re-evaluates
    // (e.g. serverHealth changes), so constructing the demo controller inline
    // there would spawn a new instance — and orphaned audio players — each time.
    @State private var demoController = DemoLaunch.makeController()
    #endif

    var body: some Scene {
        WindowGroup {
            rootView
                .environment(serverHealth)
                .preferredColorScheme(nil)        // honor system setting
                .task { await serverHealth.check() }
        }
    }

    @ViewBuilder private var rootView: some View {
        #if DEBUG
        switch DemoLaunch.screen {
        case .coach:
            NavigationStack { CoachView(controller: demoController ?? CoachController()) }
        case .coachIdle:
            NavigationStack { CoachView() }
        case .settings:
            ServerSettingsView()
        case .summary:
            ZStack {
                BreathBackdrop().ignoresSafeArea()
                ScrollView {
                    if let response = CoachController.demoResponse() {
                        SessionSummaryView(response: response).padding(20)
                    }
                }
            }
        case nil:
            HomeView()
        }
        #else
        HomeView()
        #endif
    }
}

#if DEBUG
/// Drives a screen into a populated demo state for screenshots, e.g.
/// `xcrun simctl launch <sim> <id> DEMO PHASE_PLAYING` / `PHASE_ENDED` / `SETTINGS`.
enum DemoLaunch {
    enum Screen {
        case coach(CoachController.DemoScenario)
        case coachIdle
        case settings
        case summary
    }

    static var screen: Screen? {
        let args = ProcessInfo.processInfo.arguments
        guard args.contains("DEMO") else { return nil }
        if args.contains("SETTINGS") { return .settings }
        if args.contains("SUMMARY") { return .summary }
        if args.contains("IDLE") { return .coachIdle }
        return .coach(args.contains("PHASE_ENDED") ? .ended : .playing(at: 9.0))
    }

    /// The demo controller for the `.coach` screen, or nil for other launches.
    /// Called once from the App's @State so the instance is stable.
    @MainActor static func makeController() -> CoachController? {
        guard case .coach(let scenario)? = screen else { return nil }
        return .demo(scenario)
    }
}
#endif
