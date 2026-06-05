//
//  ModelCard.swift
//  BreathCoach
//
//  Static facts about the deployed breath-detection model. The `/process`
//  response carries live per-inference numbers (params, latency); these
//  validation figures aren't in the response, so they live here as a single
//  source of truth. Update them when the deployed model changes.
//

enum ModelCard {
    /// The model these validation figures were measured on.
    static let validationVersion = "v13"

    /// Precision–recall AUC on the held-out validation set.
    static let valPRAUC = "0.65"

    /// Expected calibration error on the validation set.
    static let calibrationECE = "0.022"
}
