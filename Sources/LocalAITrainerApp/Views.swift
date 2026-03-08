import AppKit
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        TabView {
            MainDashboardView()
                .tabItem {
                    Label("Train", systemImage: "bolt.horizontal.circle")
                }

            SettingsScreen()
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
        .padding(20)
    }
}

struct MainDashboardView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                configurationCard
                environmentSummaryCard
                progressCard

                if let completedJob = appState.completedJob {
                    completionCard(job: completedJob)
                }

                previewCard
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Local AI Trainer")
                .font(.system(size: 30, weight: .semibold, design: .rounded))
            Text("Select a curated model, choose a local folder, run training, and keep the full job state on disk.")
                .foregroundStyle(.secondary)
        }
    }

    private var configurationCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Model", selection: $appState.selectedModelID) {
                    ForEach(appState.supportedModels) { model in
                        Text(model.displayName).tag(Optional(model.id))
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: appState.selectedModelID) { _, _ in
                    appState.adoptDefaultsForCurrentModel()
                }

                if let selectedModel = appState.selectedModel {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(selectedModel.subtitle)
                            .font(.headline)
                        Text(selectedModel.notes)
                            .foregroundStyle(.secondary)
                        HStack(spacing: 12) {
                            StatusChip(text: selectedModel.workflowType.displayName, color: .blue)
                            StatusChip(text: "Min RAM \(selectedModel.minMemoryGB) GB", color: .orange)
                            StatusChip(text: selectedModel.licenseLabel, color: .purple)
                            StatusChip(text: appState.modelStatusLabel(selectedModel), color: appState.modelStatusColor(selectedModel))
                            StatusChip(
                                text: appState.selectedModelBackendReady ? "Backend Ready" : (appState.isBootstrappingBackends ? "Preparing Backend" : "Backend Pending"),
                                color: appState.selectedModelBackendReady ? .green : .orange
                            )
                        }
                    }
                }

                HStack(spacing: 12) {
                    Button("Choose Dataset Folder") {
                        appState.chooseDatasetFolder()
                    }
                    .buttonStyle(.bordered)

                    Text(appState.selectedDatasetFolder?.path ?? "No folder selected")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(appState.selectedDatasetFolder == nil ? .secondary : .primary)
                        .textSelection(.enabled)
                }

                TextField("Project Name", text: $appState.projectName)
                    .textFieldStyle(.roundedBorder)

                Picker("Preset", selection: $appState.selectedPreset) {
                    ForEach(TrainingPreset.allCases) { preset in
                        Text(preset.displayName).tag(preset)
                    }
                }
                .pickerStyle(.segmented)

                Text(appState.selectedPreset.summary)
                    .foregroundStyle(.secondary)

                if let readinessMessage = appState.selectedModelReadinessMessage() {
                    Text(readinessMessage)
                        .foregroundStyle(.orange)
                }

                HStack(spacing: 12) {
                    Button("Run Training") {
                        appState.startTraining()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!appState.canStartTraining)

                    Button("Cancel Run") {
                        appState.cancelTraining()
                    }
                    .buttonStyle(.bordered)
                    .disabled(appState.canStartTraining)

                    Button("Refresh Health") {
                        Task {
                            await appState.refreshData()
                        }
                    }
                    .buttonStyle(.bordered)
                }

                if let error = appState.userFacingError {
                    Text(error)
                        .foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Training Setup", systemImage: "slider.horizontal.3")
        }
    }

    private var environmentSummaryCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    StatusChip(text: appState.environmentStatus.ollamaInstalled ? "Ollama Installed" : "Ollama Missing", color: appState.environmentStatus.ollamaInstalled ? .green : .red)
                    StatusChip(text: appState.environmentStatus.ollamaDaemonReachable ? "Daemon Ready" : "Daemon Offline", color: appState.environmentStatus.ollamaDaemonReachable ? .green : .red)
                    StatusChip(text: appState.environmentStatus.appleSilicon ? "Apple Silicon" : "Unsupported CPU", color: appState.environmentStatus.appleSilicon ? .green : .red)
                }

                HStack(spacing: 12) {
                    StatusChip(text: appState.environmentStatus.mlxLMInstalled ? "MLX-LM Ready" : "MLX-LM Missing", color: appState.environmentStatus.mlxLMInstalled ? .green : .orange)
                    StatusChip(text: appState.environmentStatus.mfluxInstalled ? "MFLUX Ready" : "MFLUX Missing", color: appState.environmentStatus.mfluxInstalled ? .green : .orange)
                    StatusChip(text: appState.environmentStatus.fullXcodeInstalled ? "Xcode Installed" : "CLT Only", color: appState.environmentStatus.fullXcodeInstalled ? .green : .orange)
                }

                Text("Memory: \(appState.environmentStatus.physicalMemoryGB) GB • Free Disk: \(appState.environmentStatus.availableDiskGB) GB")
                    .foregroundStyle(.secondary)

                if !appState.environmentStatus.blockers.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(appState.environmentStatus.blockers, id: \.self) { blocker in
                            Text("• \(blocker)")
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Environment Summary", systemImage: "externaldrive")
        }
    }

    private var progressCard: some View {
        GroupBox {
            ScrollView {
                Text(appState.runLogs.isEmpty ? "No training output yet." : appState.runLogs)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: 320, alignment: .top)
        } label: {
            Label("Progress", systemImage: "terminal")
        }
    }

    private func completionCard(job: TrainingJobManifest) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Text("Status: \(job.status.capitalized)")
                    .font(.headline)
                Text("Output Folder")
                    .font(.subheadline.weight(.medium))
                Text(job.outputDir)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)

                if let modelTag = job.ollamaModelTag {
                    Text("Local Model Tag")
                        .font(.subheadline.weight(.medium))
                    Text(modelTag)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                } else if job.workflowType == WorkflowType.image.rawValue {
                    Text("Runtime")
                        .font(.subheadline.weight(.medium))
                    Text("Direct MFLUX LoRA runtime")
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }

                if let artifactPath = job.outputArtifactPath {
                    Text("Artifact")
                        .font(.subheadline.weight(.medium))
                    Text(artifactPath)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }

                if !job.warnings.isEmpty {
                    Divider()
                    ForEach(job.warnings, id: \.self) { warning in
                        Text("• \(warning)")
                            .foregroundStyle(.orange)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Training Output", systemImage: "checkmark.seal")
        }
    }

    private var previewCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 14) {
                Text("Run the same prompt against the base model and the trained model to compare the effect of training. FLUX image models use direct MFLUX base-plus-LoRA previews, while supported text models use Ollama.")
                    .foregroundStyle(.secondary)

                TextField("Preview Prompt", text: $appState.previewPrompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)

                Button("Generate Comparison") {
                    Task {
                        await appState.generatePreview()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(appState.selectedModel == nil)

                if appState.previewState.isRunning {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView()
                        if let statusMessage = appState.previewState.statusMessage {
                            Text(statusMessage)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if let infoMessage = appState.previewState.infoMessage {
                    Text(infoMessage)
                        .foregroundStyle(.secondary)
                }

                if let error = appState.previewState.errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                }

                if appState.previewState.beforeResult != nil || appState.previewState.afterResult != nil {
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .top, spacing: 16) {
                            previewResultPanel(
                                appState.previewState.beforeResult,
                                fallbackTitle: "Before Training",
                                fallbackMessage: "Generate a preview to capture the base model output."
                            )
                            previewResultPanel(
                                appState.previewState.afterResult,
                                fallbackTitle: "After Training",
                                fallbackMessage: "Complete a training run for this model to compare the tuned output."
                            )
                        }

                        VStack(alignment: .leading, spacing: 16) {
                            previewResultPanel(
                                appState.previewState.beforeResult,
                                fallbackTitle: "Before Training",
                                fallbackMessage: "Generate a preview to capture the base model output."
                            )
                            previewResultPanel(
                                appState.previewState.afterResult,
                                fallbackTitle: "After Training",
                                fallbackMessage: "Complete a training run for this model to compare the tuned output."
                            )
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } label: {
            Label("Preview", systemImage: "photo.on.rectangle")
        }
    }

    @ViewBuilder
    private func previewResultPanel(
        _ result: PreviewResult?,
        fallbackTitle: String,
        fallbackMessage: String
    ) -> some View {
        if let result {
            PreviewResultPanel(result: result)
        } else {
            PreviewPlaceholderPanel(title: fallbackTitle, message: fallbackMessage)
        }
    }
}

struct SettingsScreen: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Settings")
                    .font(.system(size: 28, weight: .semibold, design: .rounded))

                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Install prepares everything the selected workflow needs locally: direct MFLUX base weights for FLUX image models, and both the Ollama runtime and MLX training weights for supported text models.")
                            .foregroundStyle(.secondary)

                        if appState.modelPullState.modelID != nil {
                            ModelPullProgressView(state: appState.modelPullState)
                        } else if !appState.settingsActivity.isEmpty {
                            Text(appState.settingsActivity)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                } label: {
                    Label("Catalog Policy", systemImage: "lock.shield")
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 14) {
                        ForEach(appState.supportedModels) { model in
                            HStack(alignment: .top, spacing: 16) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(model.displayName)
                                        .font(.headline)
                                    Text(model.subtitle)
                                        .foregroundStyle(.secondary)
                                    Text(model.notes)
                                        .foregroundStyle(.secondary)
                                    HStack(spacing: 8) {
                                        if model.workflowType == .image {
                                            StatusChip(text: "Direct MFLUX Runtime", color: .blue)
                                            StatusChip(
                                                text: appState.modelTrainingAssetsReady(model) ? "Base Weights Ready" : "Base Weights Needed",
                                                color: appState.modelTrainingAssetsReady(model) ? .green : .orange
                                            )
                                        } else {
                                            StatusChip(
                                                text: appState.modelRuntimeIsInstalled(model) ? "Ollama Runtime Ready" : "Ollama Runtime Missing",
                                                color: appState.modelRuntimeIsInstalled(model) ? .green : .orange
                                            )
                                            StatusChip(
                                                text: appState.modelTrainingAssetsReady(model) ? "Training Weights Ready" : "Training Weights Needed",
                                                color: appState.modelTrainingAssetsReady(model) ? .green : .orange
                                            )
                                        }
                                    }
                                }

                                Spacer()

                                VStack(alignment: .trailing, spacing: 8) {
                                    StatusChip(
                                        text: appState.modelStatusLabel(model),
                                        color: appState.modelStatusColor(model)
                                    )
                                    Button(appState.modelInstallButtonLabel(model)) {
                                        appState.pullModel(model)
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(appState.isPullingAnyModel)
                                }
                            }

                            if model.id != appState.supportedModels.last?.id {
                                Divider()
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                } label: {
                    Label("Supported Models", systemImage: "shippingbox")
                }

                GroupBox {
                    if appState.localModels.isEmpty {
                        Text("No local Ollama models were detected.")
                            .foregroundStyle(.secondary)
                    } else {
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(appState.localModels) { model in
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(model.name)
                                            .font(.headline)
                                        Text(model.details?.parameterSize ?? model.details?.family ?? "Local model")
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                }
                                if model.id != appState.localModels.last?.id {
                                    Divider()
                                }
                            }
                        }
                    }
                } label: {
                    Label("Detected Ollama Models", systemImage: "internaldrive")
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Backend Runtime")
                            .font(.headline)
                        Text("The app installs and updates the training backends automatically when they are missing.")
                            .foregroundStyle(.secondary)

                        HStack(spacing: 12) {
                            StatusChip(
                                text: appState.environmentStatus.mlxLMInstalled ? "MLX-LM Ready" : "MLX-LM Missing",
                                color: appState.environmentStatus.mlxLMInstalled ? .green : .orange
                            )
                            StatusChip(
                                text: appState.environmentStatus.mfluxInstalled ? "MFLUX Ready" : "MFLUX Missing",
                                color: appState.environmentStatus.mfluxInstalled ? .green : .orange
                            )
                            StatusChip(
                                text: appState.hasStoredHuggingFaceToken ? "HF Token Saved" : "HF Token Optional",
                                color: appState.hasStoredHuggingFaceToken ? .green : .orange
                            )
                        }

                        Button(appState.isBootstrappingBackends ? "Installing..." : "Reinstall Backends") {
                            appState.reinstallBackends()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(appState.isBootstrappingBackends)

                        if appState.isBootstrappingBackends {
                            VStack(alignment: .leading, spacing: 8) {
                                ProgressView()
                                    .controlSize(.large)
                                Text("Installing required training backends. This only happens the first time a machine is prepared.")
                                    .foregroundStyle(.secondary)
                            }
                        } else if appState.environmentStatus.mlxLMInstalled && appState.environmentStatus.mfluxInstalled {
                            Text("Required training backends are installed on this device.")
                                .foregroundStyle(.secondary)
                        } else if let backendStatus = appState.backendSetupLog
                            .split(whereSeparator: \.isNewline)
                            .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
                            .last(where: { !$0.isEmpty }) {
                            Text(backendStatus)
                                .foregroundStyle(.orange)
                        }

                        Divider()

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Hugging Face Access Token")
                                .font(.headline)
                            Text("Optional but recommended for higher rate limits and gated model access. The app stores this token in your login keychain and passes it to local training workers as `HF_TOKEN`.")
                                .foregroundStyle(.secondary)

                            SecureField("hf_...", text: $appState.huggingFaceTokenInput)
                                .textFieldStyle(.roundedBorder)

                            HStack(spacing: 12) {
                                Button(appState.hasStoredHuggingFaceToken ? "Update Token" : "Save Token") {
                                    appState.saveHuggingFaceToken()
                                }
                                .buttonStyle(.borderedProminent)
                                .disabled(appState.huggingFaceTokenInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                                Button("Remove Token") {
                                    appState.clearHuggingFaceToken()
                                }
                                .buttonStyle(.bordered)
                                .disabled(!appState.hasStoredHuggingFaceToken)
                            }

                            Text(appState.hasStoredHuggingFaceToken
                                 ? "A Hugging Face token is currently stored for local worker commands."
                                 : "No Hugging Face token is stored yet.")
                                .foregroundStyle(.secondary)

                            if let tokenStatus = appState.huggingFaceTokenStatusMessage, !tokenStatus.isEmpty {
                                Text(tokenStatus)
                                    .foregroundStyle(tokenStatus.contains("Saved") || tokenStatus.contains("Removed") ? Color.secondary : Color.orange)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                } label: {
                    Label("Backend Setup", systemImage: "wrench.and.screwdriver")
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct ModelPullProgressView: View {
    let state: ModelPullState

    private var progressColor: Color {
        if state.errorMessage != nil {
            return .red
        }
        return state.didSucceed ? .green : .blue
    }

    private var progressLabel: String {
        "\(Int((state.progressFraction ?? state.visualFraction) * 100))%"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(state.statusText)
                        .font(.headline)

                    if let modelTag = state.modelTag, !modelTag.isEmpty {
                        Text(modelTag)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                Text(progressLabel)
                    .font(.system(.caption, design: .rounded).weight(.semibold))
                    .foregroundStyle(progressColor)
            }

            ProgressView(value: min(max(state.visualFraction, 0), 1), total: 1)
                .progressViewStyle(.linear)
                .tint(progressColor)
                .controlSize(.large)

            if !state.detailText.isEmpty {
                Text(state.detailText)
                    .foregroundStyle(.secondary)
            }

            if let errorMessage = state.errorMessage {
                Text(errorMessage)
                    .foregroundStyle(.red)
            }
        }
        .padding(14)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

struct StatusChip: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.system(.caption, design: .rounded))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(color.opacity(0.14), in: Capsule())
            .foregroundStyle(color)
    }
}

struct PreviewImageView: View {
    let imagePath: String

    var body: some View {
        if let image = NSImage(contentsOfFile: imagePath) {
            Image(nsImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 360)
                .clipShape(RoundedRectangle(cornerRadius: 14))
        } else {
            Text("Preview image could not be loaded.")
                .foregroundStyle(.secondary)
        }
    }
}

struct PreviewResultPanel: View {
    let result: PreviewResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(result.title)
                .font(.headline)

            Text(result.modelTag)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            if let textOutput = result.textOutput {
                ScrollView {
                    Text(textOutput)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 180, maxHeight: 240)
            }

            if let imagePath = result.imagePath {
                PreviewImageView(imagePath: imagePath)
                Text(imagePath)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .padding(14)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

struct PreviewPlaceholderPanel: View {
    let title: String
    let message: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)

            Text(message)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 220, alignment: .topLeading)
        .padding(14)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}
