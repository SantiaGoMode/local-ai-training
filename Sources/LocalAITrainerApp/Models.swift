import Foundation

enum WorkflowType: String, Codable, CaseIterable, Identifiable {
    case image
    case text

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .image:
            return "Image Training"
        case .text:
            return "Text Training"
        }
    }
}

enum PackagingStrategy: String, Codable {
    case ollamaAdapter
    case experimentalOllamaAdapter
    case directMfluxLora

    var description: String {
        switch self {
        case .ollamaAdapter:
            return "Ollama Adapter Import"
        case .experimentalOllamaAdapter:
            return "Experimental Ollama Packaging"
        case .directMfluxLora:
            return "Direct MFLUX LoRA Runtime"
        }
    }
}

enum TrainingPreset: String, Codable, CaseIterable, Identifiable {
    case fast
    case balanced
    case quality

    var id: String { rawValue }

    var displayName: String {
        rawValue.capitalized
    }

    var summary: String {
        switch self {
        case .fast:
            return "Quick validation run with smaller training budgets."
        case .balanced:
            return "Default beta preset for general use."
        case .quality:
            return "Longer run tuned for higher adaptation quality."
        }
    }
}

struct SupportedModel: Identifiable, Hashable {
    let id: String
    let displayName: String
    let workflowType: WorkflowType
    let trainingBackendRef: String
    let ollamaRuntimeTag: String
    let packagingStrategy: PackagingStrategy
    let minMemoryGB: Int
    let licenseLabel: String
    let notes: String

    var installReference: String {
        switch packagingStrategy {
        case .directMfluxLora:
            return trainingBackendRef
        case .ollamaAdapter, .experimentalOllamaAdapter:
            return ollamaRuntimeTag
        }
    }

    var requiresOllamaRuntime: Bool {
        switch packagingStrategy {
        case .directMfluxLora:
            return false
        case .ollamaAdapter, .experimentalOllamaAdapter:
            return true
        }
    }

    var subtitle: String {
        "\(installReference) • \(packagingStrategy.description)"
    }
}

enum SupportedModelRegistry {
    static let curatedCatalog: [SupportedModel] = [
        SupportedModel(
            id: "flux2-klein-4b",
            displayName: "FLUX.2 klein 4B",
            workflowType: .image,
            trainingBackendRef: "flux2-klein-base-4b",
            ollamaRuntimeTag: "x/flux2-klein:4b",
            packagingStrategy: .directMfluxLora,
            minMemoryGB: 24,
            licenseLabel: "Non-commercial",
            notes: "Primary image workflow candidate for M4 Max machines, previewed directly through MFLUX with the trained LoRA adapter."
        ),
        SupportedModel(
            id: "flux2-klein-9b",
            displayName: "FLUX.2 klein 9B",
            workflowType: .image,
            trainingBackendRef: "flux2-klein-base-9b",
            ollamaRuntimeTag: "x/flux2-klein:9b",
            packagingStrategy: .directMfluxLora,
            minMemoryGB: 36,
            licenseLabel: "Non-commercial",
            notes: "High-memory image workflow target, previewed directly through MFLUX with the trained LoRA adapter."
        ),
        SupportedModel(
            id: "llama3.1-8b",
            displayName: "Llama 3.1 8B Instruct",
            workflowType: .text,
            trainingBackendRef: "meta-llama/Llama-3.1-8B-Instruct",
            ollamaRuntimeTag: "llama3.1:8b",
            packagingStrategy: .ollamaAdapter,
            minMemoryGB: 24,
            licenseLabel: "Community",
            notes: "General-purpose text fine-tuning option via MLX-LM and Ollama adapters."
        ),
        SupportedModel(
            id: "gemma2-9b",
            displayName: "Gemma 2 9B IT",
            workflowType: .text,
            trainingBackendRef: "google/gemma-2-9b-it",
            ollamaRuntimeTag: "gemma2:9b",
            packagingStrategy: .ollamaAdapter,
            minMemoryGB: 24,
            licenseLabel: "Gemma",
            notes: "Google Gemma instruction model for text adaptation."
        ),
        SupportedModel(
            id: "mistral-7b",
            displayName: "Mistral 7B Instruct",
            workflowType: .text,
            trainingBackendRef: "mistralai/Mistral-7B-Instruct-v0.3",
            ollamaRuntimeTag: "mistral:7b",
            packagingStrategy: .ollamaAdapter,
            minMemoryGB: 20,
            licenseLabel: "Apache 2.0",
            notes: "Smaller text workflow option with a lower memory target."
        ),
        SupportedModel(
            id: "phi3-mini",
            displayName: "Phi-3 Mini",
            workflowType: .text,
            trainingBackendRef: "microsoft/Phi-3-mini-4k-instruct",
            ollamaRuntimeTag: "phi3:latest",
            packagingStrategy: .ollamaAdapter,
            minMemoryGB: 16,
            licenseLabel: "MIT",
            notes: "Compact Phi-3 text workflow option using Ollama adapter import."
        ),
    ]
}

struct EnvironmentStatus {
    var ollamaInstalled = false
    var ollamaDaemonReachable = false
    var pythonAvailable = false
    var appleSilicon = false
    var physicalMemoryGB = 0
    var availableDiskGB = 0
    var mfluxInstalled = false
    var mlxLMInstalled = false
    var fullXcodeInstalled = false

    var blockers: [String] {
        var issues: [String] = []
        if !ollamaInstalled {
            issues.append("Ollama is not installed.")
        } else if !ollamaDaemonReachable {
            issues.append("Ollama is installed but the local daemon is not reachable.")
        }
        if !pythonAvailable {
            issues.append("Python 3 is not available on PATH.")
        }
        if !appleSilicon {
            issues.append("This app is designed for Apple Silicon Macs.")
        }
        return issues
    }
}

struct OllamaModelDetails: Decodable, Hashable {
    let family: String?
    let parameterSize: String?
}

struct OllamaLocalModel: Identifiable, Decodable, Hashable {
    let name: String
    let model: String
    let modifiedAt: String?
    let size: Int64?
    let details: OllamaModelDetails?

    var id: String { model }
}

struct OllamaPullProgress: Decodable {
    let status: String
    let digest: String?
    let total: Int64?
    let completed: Int64?
    let error: String?
}

struct ModelPullState {
    var modelID: String?
    var modelTag: String?
    var statusText = ""
    var detailText = ""
    var progressFraction: Double?
    var visualFraction = 0.0
    var isRunning = false
    var didSucceed = false
    var errorMessage: String?

    static let idle = ModelPullState()
}

struct TrainingAssetStatus: Decodable, Hashable {
    let workflow: String
    let backendRef: String
    let managed: Bool
    let ready: Bool
    let requiresDownload: Bool
    let source: String
    let repoID: String?
    let localPath: String?
    let detail: String?
}

indirect enum JSONValue: Codable, Hashable {
    case string(String)
    case integer(Int)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value.")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value):
            try container.encode(value)
        case let .integer(value):
            try container.encode(value)
        case let .number(value):
            try container.encode(value)
        case let .bool(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        case let .array(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

struct TrainingJobManifest: Codable, Identifiable {
    let jobId: String
    let projectName: String
    let workflowType: String
    let baseModelId: String
    let baseModelTag: String
    let trainingBackendRef: String
    let datasetPath: String
    let outputDir: String
    let preset: String
    let packagingStrategy: String
    let createdAt: String
    var status: String
    let logsPath: String
    var derivedDatasetPath: String?
    var outputArtifactPath: String?
    var ollamaModelTag: String?
    var errorMessage: String?
    var warnings: [String]
    var previewType: String?
    var previewPrompt: String?
    var previewOutputPath: String?
    var metadata: [String: JSONValue]

    var id: String { jobId }
}

struct PreviewResult {
    let title: String
    let modelTag: String
    let textOutput: String?
    let imagePath: String?
}

struct PreviewState {
    var isRunning = false
    var statusMessage: String?
    var beforeResult: PreviewResult?
    var afterResult: PreviewResult?
    var infoMessage: String?
    var errorMessage: String?
}

struct PreviewImageResponse: Decodable {
    let imagePath: String
}
