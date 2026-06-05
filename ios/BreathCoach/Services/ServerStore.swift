//
//  ServerStore.swift
//  BreathCoach
//
//  Persistence for the model-server address. The user sets this in Settings so
//  the app survives Wi-Fi changes without a rebuild. Plain (non-actor-isolated)
//  so the networking layer can read it from any thread; UserDefaults is
//  thread-safe.
//

import Foundation

enum ServerStore {
    private static let defaultsKey = "serverURLString"

    /// The Mac's local IP — used until the user overrides it in Settings.
    /// `serve.py` binds to all interfaces, so this works from both the
    /// simulator and a real device on the same Wi-Fi. Update it in Settings
    /// when the Mac's IP changes.
    static let fallbackURLString = "http://192.168.12.202:8080"

    static func load() -> String {
        UserDefaults.standard.string(forKey: defaultsKey) ?? fallbackURLString
    }

    static func save(_ raw: String) {
        UserDefaults.standard.set(normalize(raw), forKey: defaultsKey)
    }

    /// The base URL the networking layer should hit right now.
    static func resolvedURL() -> URL {
        url(from: load()) ?? url(from: fallbackURLString)!
    }

    /// Parse a user-entered string into a URL, tolerating a missing scheme
    /// (e.g. `192.168.1.5:8080` → `http://192.168.1.5:8080`).
    static func url(from raw: String) -> URL? {
        let trimmed = normalize(raw)
        guard let url = URL(string: trimmed), url.host != nil else { return nil }
        return url
    }

    private static func normalize(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.contains("://") { return trimmed }
        return trimmed.isEmpty ? trimmed : "http://\(trimmed)"
    }
}
