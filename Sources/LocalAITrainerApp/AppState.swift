import AppKit
import Foundation
import SwiftUI

@MainActor
final class AppState: ObservableObject {
    private static let maximumVisibleLogCharacters = 120_000

    @Published var environmentStatus = EnvironmentStatus()
    @Published var supportedModels = SupportedModelRegistry.curatedCatalog
    @Published var localModels: [OllamaLocalModel] = []
    @Published var trainingAssetStatuses: [String: TrainingAssetStatus] = [:]
    @Published var selectedModelID: String?
    @Published var projectName = "produce-marketing"
    @Published var selectedPreset: TrainingPreset = .balanced
    @Published var selectedDatasetFolder: URL?
    @Published var runLogs = ""
    @Published var settingsActivity = ""
    @Published var modelPullState = ModelPullState.idle
    @Published var userFacingError: String?
    @Published var activeJob: TrainingJobManifest?
    @Published var completedJob: TrainingJobManifest?
    @Published var previewPrompt = ""
    @Published var previewState = PreviewState()
    @Published var isRefreshing = false
    @Published var isPullingModelID: String?
    @Published var isBootstrappingBackends = false
    @Published var backendSetupLog = ""
    @Published var huggingFaceTokenInput = ""
    @Published var hasStoredHuggingFaceToken = false
    @Published var huggingFaceTokenStatusMessage: String?

    private var trainingProcess: Process?
    private var backendBootstrapProcess: Process?
    private var modelPullTask: Task<Void, Never>?
    private var pullLayerProgress: [String: PullLayerProgress] = [:]
    private var hasAttemptedAutomaticBootstrap = false

    var selectedModel: SupportedModel? {
        supportedModels.first(where: { $0.id == selectedModelID })
    }

    var canStartTraining: Bool {
        selectedModel != nil &&
        selectedDatasetFolder != nil &&
        !projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        trainingProcess == nil &&
        !isBootstrappingBackends &&
        selectedModelReadyForTraining
    }

    var localModelNames: Set<String> {
        Set(localModels.map(\.name))
    }

    var isPullingAnyModel: Bool {
        modelPullTask != nil
    }

    var selectedModelBackendReady: Bool {
        guard let selectedModel else { return false }
        switch selectedModel.workflowType {
        case .image:
            return environmentStatus.mfluxInstalled
        case .text:
            return environmentStatus.mlxLMInstalled
        }
    }

    var selectedModelReadyForTraining: Bool {
        guard let selectedModel else { return false }
        return selectedModelBackendReady && modelReadyForTraining(selectedModel)
    }

    func bootstrap() async {
        if selectedModelID == nil {
            selectedModelID = supportedModels.first?.id
        }
        if previewPrompt.isEmpty, let selectedModel {
            previewPrompt = defaultPreviewPrompt(for: selectedModel.workflowType)
        }
        loadHuggingFaceTokenState()
        await refreshData()
        maybeStartAutomaticBackendBootstrap()
    }

    func refreshData() async {
        isRefreshing = true
        defer { isRefreshing = false }
        await refreshEnvironment()
        await refreshLocalModels()
        await refreshTrainingAssetStatuses()
    }

    func refreshEnvironment() async {
        environmentStatus = await EnvironmentInspector().inspect()
    }

    func refreshLocalModels() async {
        do {
            localModels = try await OllamaService().listLocalModels()
                .sorted(by: { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending })
        } catch {
            settingsActivity = "Unable to load local Ollama models: \(error.localizedDescription)"
        }
    }

    func refreshTrainingAssetStatuses() async {
        var refreshed: [String: TrainingAssetStatus] = [:]
        let service = TrainingAssetService()

        for model in supportedModels {
            let backendReady: Bool
            switch model.workflowType {
            case .image:
                backendReady = environmentStatus.mfluxInstalled
            case .text:
                backendReady = environmentStatus.mlxLMInstalled
            }

            guard backendReady else {
                refreshed[model.id] = TrainingAssetStatus(
                    workflow: model.workflowType.rawValue,
                    backendRef: model.trainingBackendRef,
                    managed: true,
                    ready: false,
                    requiresDownload: false,
                    source: "missing-runtime",
                    repoID: nil,
                    localPath: nil,
                    detail: model.workflowType == .image
                        ? "MFLUX is not installed in the worker environment."
                        : "MLX-LM is not installed in the worker environment."
                )
                continue
            }

            do {
                refreshed[model.id] = try await service.inspectAssets(for: model)
            } catch {
                refreshed[model.id] = TrainingAssetStatus(
                    workflow: model.workflowType.rawValue,
                    backendRef: model.trainingBackendRef,
                    managed: true,
                    ready: false,
                    requiresDownload: false,
                    source: "error",
                    repoID: nil,
                    localPath: nil,
                    detail: error.localizedDescription
                )
            }
        }

        trainingAssetStatuses = refreshed
    }

    func chooseDatasetFolder() {
        let panel = NSOpenPanel()
        panel.prompt = "Choose Folder"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = selectedDatasetFolder?.deletingLastPathComponent()

        if panel.runModal() == .OK {
            selectedDatasetFolder = panel.url
        }
    }

    func modelRuntimeIsInstalled(_ model: SupportedModel) -> Bool {
        if !model.requiresOllamaRuntime {
            return true
        }
        return localModelNames.contains(model.ollamaRuntimeTag)
    }

    func modelTrainingAssetsStatus(_ model: SupportedModel) -> TrainingAssetStatus? {
        trainingAssetStatuses[model.id]
    }

    func modelTrainingAssetsReady(_ model: SupportedModel) -> Bool {
        return trainingAssetStatuses[model.id]?.ready == true
    }

    func modelIsInstalled(_ model: SupportedModel) -> Bool {
        switch model.workflowType {
        case .image:
            return modelTrainingAssetsReady(model)
        case .text:
            return modelRuntimeIsInstalled(model) && modelTrainingAssetsReady(model)
        }
    }

    func modelReadyForTraining(_ model: SupportedModel) -> Bool {
        modelIsInstalled(model)
    }

    func modelStatusLabel(_ model: SupportedModel) -> String {
        if isPullingModelID == model.id {
            return "Installing"
        }
        if modelIsInstalled(model) {
            return "Installed"
        }
        return "Not Installed"
    }

    func modelStatusColor(_ model: SupportedModel) -> Color {
        switch modelStatusLabel(model) {
        case "Installed":
            return .green
        case "Installing":
            return .blue
        case "Runtime Only":
            return .orange
        default:
            return .orange
        }
    }

    func modelInstallButtonLabel(_ model: SupportedModel) -> String {
        if isPullingModelID == model.id {
            return "Installing..."
        }
        if modelIsInstalled(model) {
            return "Reinstall"
        }
        return "Install"
    }

    func selectedModelReadinessMessage() -> String? {
        guard let selectedModel else { return nil }
        if !selectedModelBackendReady {
            return selectedModel.workflowType == .image
                ? "The app is installing the required MFLUX runtime automatically."
                : "The app is installing the required MLX-LM runtime automatically."
        }
        if selectedModel.requiresOllamaRuntime && !modelRuntimeIsInstalled(selectedModel) {
            return "Install this model in Settings before starting training."
        }
        if !modelTrainingAssetsReady(selectedModel) {
            return selectedModel.workflowType == .image
                ? "Install this FLUX model in Settings to prepare the MFLUX base weights before you run training."
                : "Install this model in Settings to prepare the MLX training weights before you run training."
        }
        return nil
    }

    func pullModel(_ model: SupportedModel) {
        guard modelPullTask == nil else { return }

        isPullingModelID = model.id
        pullLayerProgress = [:]
        modelPullState = ModelPullState(
            modelID: model.id,
            modelTag: model.installReference,
            statusText: "Preparing download",
            detailText: model.requiresOllamaRuntime
                ? "Checking model availability and opening a secure transfer with Ollama."
                : "Checking the local MFLUX cache and preparing the base model weights required for training and preview.",
            progressFraction: nil,
            visualFraction: 0.06,
            isRunning: true,
            didSucceed: false,
            errorMessage: nil
        )
        settingsActivity = "Preparing \(model.installReference)..."

        modelPullTask = Task { [weak self] in
            guard let self else { return }
            do {
                if model.requiresOllamaRuntime && !self.modelRuntimeIsInstalled(model) {
                    try await OllamaService().pullModel(tag: model.ollamaRuntimeTag) { update in
                        await MainActor.run {
                            self.applyPullProgress(update, for: model)
                        }
                    }
                } else if model.requiresOllamaRuntime {
                    await MainActor.run {
                        self.modelPullState = ModelPullState(
                            modelID: model.id,
                            modelTag: model.ollamaRuntimeTag,
                            statusText: "Runtime ready",
                            detailText: "The Ollama runtime is already installed locally.",
                            progressFraction: 0.55,
                            visualFraction: 0.55,
                            isRunning: true,
                            didSucceed: false,
                            errorMessage: nil
                        )
                        self.settingsActivity = "Runtime already installed for \(model.ollamaRuntimeTag)."
                    }
                }

                if model.workflowType == .image || model.workflowType == .text {
                    await MainActor.run {
                        self.modelPullState = ModelPullState(
                            modelID: model.id,
                            modelTag: model.trainingBackendRef,
                            statusText: model.workflowType == .image
                                ? "Preparing FLUX base weights"
                                : "Preparing training weights",
                            detailText: model.workflowType == .image
                                ? "Installing the FLUX base weights required by MFLUX so training and preview run directly without Ollama packaging."
                                : "Installing the MLX training weights required for local fine-tuning so the run does not fetch model files during training.",
                            progressFraction: nil,
                            visualFraction: model.requiresOllamaRuntime ? 0.78 : 0.2,
                            isRunning: true,
                            didSucceed: false,
                            errorMessage: nil
                        )
                        self.settingsActivity = model.workflowType == .image
                            ? "Preparing MFLUX assets for \(model.displayName)..."
                            : "Preparing MLX training assets for \(model.displayName)..."
                    }

                    let assetStatus = try await TrainingAssetService().prepareAssets(for: model)
                    await MainActor.run {
                        self.trainingAssetStatuses[model.id] = assetStatus
                    }
                }

                await self.finishModelPullSuccessfully(for: model)
            } catch {
                await self.finishModelPullWithError(error, for: model)
            }
        }
    }

    func reinstallBackends() {
        do {
            try launchBackendBootstrap(triggeredAutomatically: false)
        } catch {
            backendSetupLog = error.localizedDescription
        }
    }

    func saveHuggingFaceToken() {
        do {
            try HuggingFaceTokenStore.saveToken(huggingFaceTokenInput)
            huggingFaceTokenInput = ""
            hasStoredHuggingFaceToken = true
            huggingFaceTokenStatusMessage = "Saved the Hugging Face token in the login keychain."
        } catch {
            huggingFaceTokenStatusMessage = error.localizedDescription
        }
    }

    func clearHuggingFaceToken() {
        do {
            try HuggingFaceTokenStore.clearToken()
            huggingFaceTokenInput = ""
            hasStoredHuggingFaceToken = false
            huggingFaceTokenStatusMessage = "Removed the Hugging Face token from the login keychain."
        } catch {
            huggingFaceTokenStatusMessage = error.localizedDescription
        }
    }

    func startTraining() {
        guard let selectedModel else {
            userFacingError = "Choose a supported model before starting training."
            return
        }
        guard let selectedDatasetFolder else {
            userFacingError = "Choose a dataset folder before starting training."
            return
        }
        guard modelReadyForTraining(selectedModel) else {
            userFacingError = selectedModelReadinessMessage() ?? "Install the selected model and its training assets before starting training."
            return
        }

        do {
            try AppDirectories.ensureBaseDirectories()
            let jobID = Self.makeJobID()
            let jobDirectory = AppDirectories.runsDirectory.appendingPathComponent(jobID, isDirectory: true)
            try FileManager.default.createDirectory(at: jobDirectory, withIntermediateDirectories: true)

            var metadata: [String: JSONValue] = [
                "licenseLabel": .string(selectedModel.licenseLabel),
                "notes": .string(selectedModel.notes),
            ]
            if let localPath = modelTrainingAssetsStatus(selectedModel)?.localPath {
                metadata["trainingModelPath"] = .string(localPath)
            }

            let manifest = TrainingJobManifest(
                jobId: jobID,
                projectName: projectName.trimmingCharacters(in: .whitespacesAndNewlines),
                workflowType: selectedModel.workflowType.rawValue,
                baseModelId: selectedModel.id,
                baseModelTag: selectedModel.installReference,
                trainingBackendRef: selectedModel.trainingBackendRef,
                datasetPath: selectedDatasetFolder.path,
                outputDir: jobDirectory.path,
                preset: selectedPreset.rawValue,
                packagingStrategy: selectedModel.packagingStrategy.rawValue,
                createdAt: ISO8601DateFormatter().string(from: Date()),
                status: "queued",
                logsPath: jobDirectory.appendingPathComponent("worker.log").path,
                derivedDatasetPath: nil,
                outputArtifactPath: nil,
                ollamaModelTag: nil,
                errorMessage: nil,
                warnings: [],
                previewType: selectedModel.workflowType.rawValue,
                previewPrompt: defaultPreviewPrompt(for: selectedModel.workflowType),
                previewOutputPath: nil,
                metadata: metadata
            )

            let manifestURL = jobDirectory.appendingPathComponent("manifest.json")
            try persistManifest(manifest, to: manifestURL)

            runLogs = "Queued training job \(jobID)\n"
            userFacingError = nil
            previewState = PreviewState()
            previewPrompt = defaultPreviewPrompt(for: selectedModel.workflowType)
            completedJob = nil
            activeJob = manifest

            try launchWorkerProcess(manifestURL: manifestURL)
        } catch {
            userFacingError = error.localizedDescription
        }
    }

    func cancelTraining() {
        appendRunLog("Cancellation requested.\n")
        trainingProcess?.terminate()
    }

    func generatePreview() async {
        guard let selectedModel else {
            previewState.errorMessage = "Select a model before generating a preview."
            return
        }

        let beforeReference = selectedModel.workflowType == .image
            ? selectedModel.trainingBackendRef
            : selectedModel.ollamaRuntimeTag
        let afterModelTag = trainedPreviewModelTag(for: selectedModel)
        let afterArtifactPath = trainedPreviewArtifactPath(for: selectedModel)
        let imageModelPath = selectedModel.workflowType == .image
            ? modelTrainingAssetsStatus(selectedModel)?.localPath
            : nil

        if selectedModel.workflowType == .image && imageModelPath == nil {
            previewState.errorMessage = "Finish installing this FLUX model in Settings before generating a comparison."
            return
        }

        let missingAfterMessage: String?
        if selectedModel.workflowType == .image {
            missingAfterMessage = afterArtifactPath == nil
                ? "Run training for this model to compare the saved FLUX LoRA adapter against the base model."
                : nil
        } else {
            missingAfterMessage = afterModelTag == nil
                ? "Run training for this model to unlock the after-training comparison."
                : nil
        }

        previewState = PreviewState(
            isRunning: true,
            statusMessage: "Generating before-training preview...",
            beforeResult: nil,
            afterResult: nil,
            infoMessage: missingAfterMessage,
            errorMessage: nil
        )

        do {
            switch selectedModel.workflowType {
            case .text:
                let beforeResponse = try await OllamaService().generateText(model: beforeReference, prompt: previewPrompt)
                previewState.beforeResult = PreviewResult(
                    title: "Before Training",
                    modelTag: beforeReference,
                    textOutput: beforeResponse,
                    imagePath: nil
                )

                if let afterModelTag {
                    previewState.statusMessage = "Generating after-training preview..."
                    let afterResponse = try await OllamaService().generateText(model: afterModelTag, prompt: previewPrompt)
                    previewState.afterResult = PreviewResult(
                        title: "After Training",
                        modelTag: afterModelTag,
                        textOutput: afterResponse,
                        imagePath: nil
                    )
                }
            case .image:
                let workerScript = try WorkerLocator.scriptURL()
                let pythonInterpreter = PythonLocator.interpreterURL()
                let beforeImagePath = try await generateFluxImagePreview(
                    workerScript: workerScript,
                    pythonInterpreter: pythonInterpreter,
                    backendRef: selectedModel.trainingBackendRef,
                    modelPath: imageModelPath,
                    loraPath: nil,
                    previewRole: "before"
                )
                previewState.beforeResult = PreviewResult(
                    title: "Before Training (Base)",
                    modelTag: beforeReference,
                    textOutput: nil,
                    imagePath: beforeImagePath
                )

                if let afterArtifactPath {
                    previewState.statusMessage = "Generating after-training preview..."
                    let afterImagePath = try await generateFluxImagePreview(
                        workerScript: workerScript,
                        pythonInterpreter: pythonInterpreter,
                        backendRef: selectedModel.trainingBackendRef,
                        modelPath: imageModelPath,
                        loraPath: afterArtifactPath,
                        previewRole: "after"
                    )
                    previewState.afterResult = PreviewResult(
                        title: "After Training (LoRA)",
                        modelTag: "\(selectedModel.trainingBackendRef) + \(URL(fileURLWithPath: afterArtifactPath).lastPathComponent)",
                        textOutput: nil,
                        imagePath: afterImagePath
                    )
                }
            }

            previewState.isRunning = false
            previewState.statusMessage = nil
        } catch {
            previewState.isRunning = false
            previewState.statusMessage = nil
            previewState.errorMessage = error.localizedDescription
        }
    }

    func adoptDefaultsForCurrentModel() {
        if let selectedModel {
            previewPrompt = defaultPreviewPrompt(for: selectedModel.workflowType)
        }
    }

    private func trainedPreviewModelTag(for model: SupportedModel) -> String? {
        guard let completedJob else { return nil }
        guard completedJob.status == "completed", completedJob.baseModelId == model.id else { return nil }
        guard let modelTag = completedJob.ollamaModelTag, !modelTag.isEmpty else { return nil }
        return modelTag
    }

    private func trainedPreviewArtifactPath(for model: SupportedModel) -> String? {
        guard let completedJob else { return nil }
        guard completedJob.status == "completed", completedJob.baseModelId == model.id else { return nil }
        guard let artifactPath = completedJob.outputArtifactPath, !artifactPath.isEmpty else { return nil }
        return artifactPath
    }

    private func generateFluxImagePreview(
        workerScript: URL,
        pythonInterpreter: URL,
        backendRef: String,
        modelPath: String?,
        loraPath: String?,
        previewRole: String
    ) async throws -> String {
        let outputDirectory = AppDirectories.previewsDirectory
            .appendingPathComponent(previewRole, isDirectory: true)
            .appendingPathComponent(backendRef.replacingOccurrences(of: "/", with: "_"), isDirectory: true)
        var arguments = [
            workerScript.path,
            "preview-image",
            "--backend-ref",
            backendRef,
            "--prompt",
            previewPrompt,
            "--output-dir",
            outputDirectory.path,
        ]
        if let modelPath {
            arguments.append(contentsOf: ["--model-path", modelPath])
        }
        if let loraPath {
            arguments.append(contentsOf: ["--lora-path", loraPath])
        }
        let result = try await ShellCommandRunner().run(
            executable: pythonInterpreter.path,
            arguments: arguments,
            currentDirectory: workerScript.deletingLastPathComponent().deletingLastPathComponent(),
            environment: WorkerProcessEnvironment.current()
        )
        guard result.exitCode == 0 else {
            throw NSError(
                domain: "LocalAITrainer.Preview",
                code: Int(result.exitCode),
                userInfo: [
                    NSLocalizedDescriptionKey: sanitizedPreviewCommandOutput(result.output)
                ]
            )
        }
        let payload = try decodePreviewImageResponse(from: result.output)
        return payload.imagePath
    }

    private func decodePreviewImageResponse(from output: String) throws -> PreviewImageResponse {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        if let payload = try? decoder.decode(PreviewImageResponse.self, from: Data(output.utf8)) {
            return payload
        }

        let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if let startIndex = trimmedOutput.lastIndex(of: "{") {
            let candidate = String(trimmedOutput[startIndex...])
            if let payload = try? decoder.decode(PreviewImageResponse.self, from: Data(candidate.utf8)) {
                return payload
            }
        }

        throw NSError(
            domain: "LocalAITrainer.Preview",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey: sanitizedPreviewCommandOutput(output)
            ]
        )
    }

    private func sanitizedPreviewCommandOutput(_ output: String) -> String {
        let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedOutput.isEmpty else {
            return "The preview worker did not return a readable response."
        }

        let lines = trimmedOutput
            .split(separator: "\n", omittingEmptySubsequences: true)
            .suffix(8)
            .map(String.init)

        return lines.joined(separator: "\n")
    }

    private func maybeStartAutomaticBackendBootstrap() {
        guard !environmentStatus.mlxLMInstalled || !environmentStatus.mfluxInstalled else { return }
        guard environmentStatus.pythonAvailable else {
            backendSetupLog = "Python 3 is required before the app can install its backend runtimes automatically."
            return
        }
        guard !hasAttemptedAutomaticBootstrap else { return }
        guard backendBootstrapProcess == nil else { return }

        hasAttemptedAutomaticBootstrap = true
        do {
            try launchBackendBootstrap(triggeredAutomatically: true)
        } catch {
            backendSetupLog = error.localizedDescription
        }
    }

    private func launchBackendBootstrap(triggeredAutomatically: Bool) throws {
        guard backendBootstrapProcess == nil else { return }

        let scriptURL = try BackendSetupLocator.scriptURL()
        let process = Process()
        let pipe = Pipe()

        backendSetupLog = triggeredAutomatically
            ? "Preparing training backends automatically...\n"
            : "Reinstalling training backends...\n"
        isBootstrappingBackends = true

        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [scriptURL.path]
        process.currentDirectoryURL = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in
                self?.appendBackendSetupLog(text)
            }
        }

        process.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor in
                await self?.finishBackendBootstrap(terminationStatus: process.terminationStatus)
            }
        }

        backendBootstrapProcess = process
        try process.run()
    }

    private func launchWorkerProcess(manifestURL: URL) throws {
        let workerScript = try WorkerLocator.scriptURL()
        let pythonInterpreter = PythonLocator.interpreterURL()
        let process = Process()
        let pipe = Pipe()

        process.executableURL = pythonInterpreter
        process.arguments = [workerScript.path, "run-job", "--manifest", manifestURL.path]
        process.currentDirectoryURL = workerScript.deletingLastPathComponent().deletingLastPathComponent()
        process.environment = WorkerProcessEnvironment.current()
        process.standardOutput = pipe
        process.standardError = pipe

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in
                self?.appendRunLog(text)
            }
        }

        process.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            Task { @MainActor in
                await self?.finishWorkerProcess(terminationStatus: process.terminationStatus, manifestURL: manifestURL)
            }
        }

        trainingProcess = process
        try process.run()
    }

    private func finishWorkerProcess(terminationStatus: Int32, manifestURL: URL) async {
        trainingProcess = nil

        guard let updatedManifest = try? loadManifest(from: manifestURL) else {
            userFacingError = "The worker exited, but the manifest could not be reloaded."
            return
        }

        activeJob = updatedManifest
        previewPrompt = updatedManifest.previewPrompt ?? previewPrompt

        if updatedManifest.status == "completed" {
            completedJob = updatedManifest
            userFacingError = nil
        } else if updatedManifest.status == "cancelled" {
            userFacingError = updatedManifest.errorMessage ?? "Training was cancelled."
        } else {
            userFacingError = updatedManifest.errorMessage ?? "Training failed with exit code \(terminationStatus)."
        }

        await refreshLocalModels()
    }

    private func appendRunLog(_ text: String) {
        runLogs = Self.appendingVisibleLogChunk(text, to: runLogs)
    }

    private func appendBackendSetupLog(_ text: String) {
        backendSetupLog = Self.appendingVisibleLogChunk(text, to: backendSetupLog)
    }

    private func finishBackendBootstrap(terminationStatus: Int32) async {
        backendBootstrapProcess = nil
        isBootstrappingBackends = false
        if terminationStatus == 0 {
            appendBackendSetupLog("Backend runtime is ready.\n")
        } else {
            appendBackendSetupLog("Backend bootstrap failed with exit code \(terminationStatus).\n")
        }
        await refreshEnvironment()
    }

    private static func appendingVisibleLogChunk(_ chunk: String, to existing: String) -> String {
        let combined = existing.isEmpty ? chunk : existing + chunk
        guard combined.count > maximumVisibleLogCharacters else {
            return combined
        }

        let suffix = String(combined.suffix(maximumVisibleLogCharacters))
        if let newlineIndex = suffix.firstIndex(of: "\n") {
            return String(suffix[suffix.index(after: newlineIndex)...])
        }
        return suffix
    }

    private func applyPullProgress(_ update: OllamaPullProgress, for model: SupportedModel) {
        if let digest = update.digest, let total = update.total, total > 0 {
            let completed = min(update.completed ?? 0, total)
            pullLayerProgress[digest] = PullLayerProgress(completed: completed, total: total)
        }

        let aggregate = aggregatePullProgress()
        let statusText = Self.userFacingPullStatus(for: update)
        let detailText = Self.userFacingPullDetail(for: update, aggregate: aggregate)
        let actualFraction = aggregate.map { $0.fraction }
        let visualFraction = Self.visualPullFraction(for: update.status, actualFraction: actualFraction)

        modelPullState = ModelPullState(
            modelID: model.id,
            modelTag: model.ollamaRuntimeTag,
            statusText: statusText,
            detailText: detailText,
            progressFraction: actualFraction,
            visualFraction: visualFraction,
            isRunning: true,
            didSucceed: false,
            errorMessage: nil
        )
        settingsActivity = statusText
    }

    private func finishModelPullSuccessfully(for model: SupportedModel) async {
        modelPullTask = nil
        isPullingModelID = nil
        await refreshLocalModels()
        await refreshTrainingAssetStatuses()

        modelPullState = ModelPullState(
            modelID: model.id,
            modelTag: model.installReference,
            statusText: "Model installed",
            detailText: model.workflowType == .image
                ? "\(model.displayName) is ready for direct MFLUX training and comparison previews."
                : "\(model.displayName) is ready and available on the training screen.",
            progressFraction: 1.0,
            visualFraction: 1.0,
            isRunning: false,
            didSucceed: true,
            errorMessage: nil
        )
        settingsActivity = model.workflowType == .image
            ? "Prepared MFLUX weights for \(model.displayName)."
            : "Pulled \(model.ollamaRuntimeTag). Model is ready."
    }

    private func finishModelPullWithError(_ error: Error, for model: SupportedModel) async {
        modelPullTask = nil
        isPullingModelID = nil
        await refreshLocalModels()
        await refreshTrainingAssetStatuses()

        let message = error.localizedDescription
        let isAvailableLocally = modelIsInstalled(model)

        if isAvailableLocally {
            modelPullState = ModelPullState(
                modelID: model.id,
                modelTag: model.installReference,
                statusText: "Model available locally",
                detailText: model.workflowType == .image
                    ? "The required FLUX base weights are already available locally."
                    : "Ollama reports the model is installed, even though the final pull request returned an error.",
                progressFraction: 1.0,
                visualFraction: 1.0,
                isRunning: false,
                didSucceed: true,
                errorMessage: nil
            )
            settingsActivity = model.workflowType == .image
                ? "\(model.displayName) base weights are already available locally."
                : "\(model.ollamaRuntimeTag) is already available locally."
            return
        }

        modelPullState = ModelPullState(
            modelID: model.id,
            modelTag: model.installReference,
            statusText: "Installation failed",
            detailText: model.workflowType == .image
                ? "The FLUX base-weight preparation did not complete. Review the message below and try again."
                : "The model download did not complete. Review the message below and try again.",
            progressFraction: modelPullState.progressFraction,
            visualFraction: modelPullState.visualFraction,
            isRunning: false,
            didSucceed: false,
            errorMessage: message
        )
        settingsActivity = message
    }

    private func aggregatePullProgress() -> PullProgressAggregate? {
        let totals = pullLayerProgress.values.reduce(into: (completed: Int64(0), total: Int64(0))) { partialResult, layer in
            partialResult.completed += layer.completed
            partialResult.total += layer.total
        }
        guard totals.total > 0 else { return nil }
        let fraction = min(max(Double(totals.completed) / Double(totals.total), 0), 1)
        return PullProgressAggregate(completed: totals.completed, total: totals.total, fraction: fraction)
    }

    private func persistManifest(_ manifest: TrainingJobManifest, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: url)
    }

    private func loadManifest(from url: URL) throws -> TrainingJobManifest {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TrainingJobManifest.self, from: Data(contentsOf: url))
    }

    private func defaultPreviewPrompt(for workflow: WorkflowType) -> String {
        switch workflow {
        case .image:
            return "Create a polished Instagram-ready produce advertisement with fresh colors and clean lighting."
        case .text:
            return "Write a concise social post for a new organic produce launch."
        }
    }

    private func loadHuggingFaceTokenState() {
        if let token = try? HuggingFaceTokenStore.readToken(), !token.isEmpty {
            hasStoredHuggingFaceToken = true
        } else {
            hasStoredHuggingFaceToken = false
        }
        huggingFaceTokenInput = ""
        huggingFaceTokenStatusMessage = nil
    }

    private static func makeJobID() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "\(formatter.string(from: Date()))-\(UUID().uuidString.prefix(8).lowercased())"
    }

    private static func userFacingPullStatus(for update: OllamaPullProgress) -> String {
        let lowercased = update.status.lowercased()
        if lowercased.contains("manifest") && lowercased.contains("pulling") {
            return "Preparing model files"
        }
        if lowercased.contains("verifying") {
            return "Verifying download"
        }
        if lowercased.contains("writing manifest") {
            return "Finalizing installation"
        }
        if lowercased == "success" {
            return "Model installed"
        }
        if update.digest != nil {
            return "Downloading model files"
        }
        return update.status.capitalized
    }

    private static func userFacingPullDetail(for update: OllamaPullProgress, aggregate: PullProgressAggregate?) -> String {
        let lowercased = update.status.lowercased()

        if let aggregate {
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file
            let completed = formatter.string(fromByteCount: aggregate.completed)
            let total = formatter.string(fromByteCount: aggregate.total)
            return "\(Int(aggregate.fraction * 100))% complete • \(completed) of \(total)"
        }

        if lowercased.contains("manifest") && lowercased.contains("pulling") {
            return "Checking which model layers need to be downloaded."
        }
        if lowercased.contains("verifying") {
            return "Validating downloaded files before the install is marked ready."
        }
        if lowercased.contains("writing manifest") {
            return "Saving the model metadata so it appears in the local catalog."
        }
        if lowercased == "success" {
            return "Installation completed successfully."
        }
        return "This can take a few minutes for larger models."
    }

    private static func visualPullFraction(for status: String, actualFraction: Double?) -> Double {
        let lowercased = status.lowercased()
        let stageMinimum: Double

        if lowercased.contains("manifest") && lowercased.contains("pulling") {
            stageMinimum = 0.06
        } else if lowercased.contains("verifying") {
            stageMinimum = 0.94
        } else if lowercased.contains("writing manifest") {
            stageMinimum = 0.98
        } else if lowercased == "success" {
            stageMinimum = 1.0
        } else {
            stageMinimum = 0.12
        }

        guard let actualFraction else {
            return stageMinimum
        }
        if lowercased == "success" {
            return 1.0
        }
        return min(max(actualFraction, stageMinimum), 0.995)
    }

    private struct PullLayerProgress {
        let completed: Int64
        let total: Int64
    }

    private struct PullProgressAggregate {
        let completed: Int64
        let total: Int64
        let fraction: Double
    }
}
