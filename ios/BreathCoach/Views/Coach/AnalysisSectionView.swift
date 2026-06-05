//
//  AnalysisSectionView.swift
//  BreathCoach
//
//  Collapsible "Analysis & model details" — spectrogram, breath probability
//  curve, model meta. Custom expand/collapse instead of DisclosureGroup so we
//  control width-bounding and animation. See ui.md §4.6.
//

import SwiftUI

struct AnalysisSectionView: View {
    let response: ProcessResponse
    let currentTime: Double

    @State private var isExpanded = false

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 0) {
                header
                if isExpanded {
                    body_
                        .padding(.top, 16)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .animation(.easeInOut(duration: 0.25), value: isExpanded)
    }

    // MARK: - Header

    private var header: some View {
        Button {
            isExpanded.toggle()
        } label: {
            HStack {
                Text("Analysis & model details")
                    .font(.system(.callout, design: .rounded, weight: .semibold))
                    .foregroundStyle(Color.breathInkPrimary)
                Spacer()
                Image(systemName: "chevron.down")
                    .font(.system(.caption, weight: .bold))
                    .foregroundStyle(Color.breathInkMuted)
                    .rotationEffect(.degrees(isExpanded ? 180 : 0))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Body

    private var body_: some View {
        VStack(alignment: .leading, spacing: 18) {
            spectrogramSection
            probabilitySection
            modelMetaSection
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Spectrogram

    @ViewBuilder private var spectrogramSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(left: "Log-mel spectrogram", right: "40 bands · 10 ms hop")

            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color(red: 0.04, green: 0.04, blue: 0.07))

                if let path = response.spectrogramFile {
                    let url = BreathAPI.shared.staticFileURL(forRelativePath: path)
                    AsyncImage(url: url) { phase in
                        switch phase {
                        case .success(let image):
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                        case .failure:
                            placeholderText("spectrogram unavailable")
                        case .empty:
                            ProgressView().tint(.white)
                        @unknown default:
                            EmptyView()
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                } else {
                    placeholderText("no spectrogram for this clip")
                }

                // Event overlays — magenta bottom strip (Ruinskiy), indigo
                // top strip (BreathHead). Drawn even if image is still loading.
                Canvas { context, size in
                    drawEventBands(in: context, size: size)
                    drawPlayhead(in: context, size: size)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 160)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func drawEventBands(in context: GraphicsContext, size: CGSize) {
        guard response.durationSec > 0 else { return }
        for ev in response.predictedEvents {
            let x0 = CGFloat(ev.startSec / response.durationSec) * size.width
            let x1 = CGFloat(ev.endSec / response.durationSec) * size.width
            let rect = CGRect(x: x0, y: 2, width: max(2, x1 - x0), height: 5)
            context.fill(Path(roundedRect: rect, cornerRadius: 1.5),
                         with: .color(.breathIndigo))
        }
        for ev in response.ruinskiyEvents {
            let x0 = CGFloat(ev.startSec / response.durationSec) * size.width
            let x1 = CGFloat(ev.endSec / response.durationSec) * size.width
            let rect = CGRect(x: x0, y: size.height - 7, width: max(2, x1 - x0), height: 5)
            context.fill(Path(roundedRect: rect, cornerRadius: 1.5),
                         with: .color(.breathMagenta))
        }
    }

    private func drawPlayhead(in context: GraphicsContext, size: CGSize) {
        guard currentTime > 0, response.durationSec > 0 else { return }
        let x = CGFloat(currentTime / response.durationSec) * size.width
        var path = Path()
        path.move(to: CGPoint(x: x, y: 0))
        path.addLine(to: CGPoint(x: x, y: size.height))
        context.stroke(path, with: .color(Color.white.opacity(0.85)), lineWidth: 1)
    }

    private func placeholderText(_ text: String) -> some View {
        Text(text)
            .font(.system(.caption, design: .rounded))
            .foregroundStyle(Color.breathDim)
    }

    // MARK: - Probability curve

    @ViewBuilder private var probabilitySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(
                left: "Breath probability",
                right: String(format: "threshold %.2f", response.threshold)
            )

            ZStack {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.white.opacity(0.08))
                Canvas { context, size in
                    drawProbCurve(in: context, size: size)
                }
                .padding(.vertical, 6)
                .padding(.horizontal, 8)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 110)

            probLegend
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var probLegend: some View {
        HStack(spacing: 14) {
            legendChip(.breathIndigo, "BreathHead")
            legendChip(.breathAmber, "threshold")
            legendChip(.breathMagenta, "Ruinskiy 2007")
            Spacer(minLength: 0)
        }
        .font(.system(.caption2, design: .rounded))
        .foregroundStyle(Color.breathInkMuted)
    }

    private func drawProbCurve(in context: GraphicsContext, size: CGSize) {
        guard !response.breathProb.isEmpty, response.durationSec > 0 else { return }

        // Detected regions shaded indigo top-to-bottom.
        for ev in response.predictedEvents {
            let x0 = CGFloat(ev.startSec / response.durationSec) * size.width
            let x1 = CGFloat(ev.endSec / response.durationSec) * size.width
            let rect = CGRect(x: x0, y: 0, width: max(2, x1 - x0), height: size.height)
            context.fill(Path(rect), with: .color(Color.breathIndigo.opacity(0.20)))
        }

        // Ruinskiy bottom strip.
        for ev in response.ruinskiyEvents {
            let x0 = CGFloat(ev.startSec / response.durationSec) * size.width
            let x1 = CGFloat(ev.endSec / response.durationSec) * size.width
            let rect = CGRect(x: x0, y: size.height - 10, width: max(2, x1 - x0), height: 10)
            context.fill(Path(rect), with: .color(Color.breathMagenta.opacity(0.55)))
        }

        // Dashed amber threshold.
        let yThresh = size.height * (1 - CGFloat(response.threshold))
        var th = Path()
        th.move(to: CGPoint(x: 0, y: yThresh))
        th.addLine(to: CGPoint(x: size.width, y: yThresh))
        context.stroke(th, with: .color(.breathAmber),
                       style: StrokeStyle(lineWidth: 1, dash: [4, 3]))

        // Curve. Clamp to [0,1] and drop non-finite values — a NaN/∞ coordinate
        // in a Path crashes CoreGraphics, so never feed it raw model output.
        var curve = Path()
        let n = response.breathProb.count
        for (i, p) in response.breathProb.enumerated() {
            let clamped = p.isFinite ? min(1, max(0, p)) : 0
            let x = CGFloat(i) / CGFloat(max(1, n - 1)) * size.width
            let y = size.height * (1 - CGFloat(clamped))
            if i == 0 { curve.move(to: CGPoint(x: x, y: y)) }
            else      { curve.addLine(to: CGPoint(x: x, y: y)) }
        }
        context.stroke(curve,
                       with: .linearGradient(Gradient(colors: [.breathTeal, .breathIndigo]),
                                             startPoint: .zero,
                                             endPoint: CGPoint(x: size.width, y: 0)),
                       lineWidth: 2)

        if currentTime > 0 {
            let x = CGFloat(currentTime / response.durationSec) * size.width
            var path = Path()
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
            context.stroke(path, with: .color(Color.white.opacity(0.85)), lineWidth: 1)
        }
    }

    // MARK: - Model meta

    private var modelMetaSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionHeader(left: "Model", right: nil)
            VStack(spacing: 8) {
                // Live, per-inference figures from this clip's response.
                metaRow(label: "Architecture",
                        value: "BreathHead · \(formatParams(response.modelMeta.params))")
                metaRow(label: "Latency",
                        value: response.modelMeta.perFrameMs
                            .map { String(format: "%.2f ms / frame", $0) } ?? "—")
                // Static validation figures (not in the response — see ModelCard).
                metaRow(label: "Val PR-AUC", value: ModelCard.valPRAUC)
                metaRow(label: "Calibration (ECE)", value: ModelCard.calibrationECE)
            }
            Text("PR-AUC and ECE are validation figures for BreathHead \(ModelCard.validationVersion).")
                .font(.system(.caption2, design: .rounded))
                .foregroundStyle(Color.breathInkMuted)
                .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func metaRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(.caption, design: .rounded))
                .foregroundStyle(Color.breathInkMuted)
            Spacer()
            Text(value)
                .font(.system(.callout, design: .rounded, weight: .semibold).monospacedDigit())
                .foregroundStyle(Color.breathInkPrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.06))
        )
    }

    private func formatParams(_ n: Int?) -> String {
        guard let n else { return "—" }
        if n >= 1000 { return String(format: "%.1fk params", Double(n)/1000.0) }
        return "\(n) params"
    }

    // MARK: - Helpers

    private func sectionHeader(left: String, right: String?) -> some View {
        HStack {
            Text(left.uppercased())
                .font(.system(.caption2, design: .rounded, weight: .semibold))
                .tracking(1.5)
            Spacer()
            if let right {
                Text(right.uppercased())
                    .font(.system(.caption2, design: .rounded, weight: .semibold))
                    .tracking(1.5)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
        }
        .foregroundStyle(Color.breathInkMuted)
    }

    private func legendChip(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 6) {
            Capsule().fill(color).frame(width: 16, height: 4)
            Text(label).lineLimit(1)
        }
    }
}
