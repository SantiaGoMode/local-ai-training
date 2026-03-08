import Foundation
import Security

struct ShellCommandResult {
    let exitCode: Int32
    let output: String
}

enum AppDirectories {
    static var appSupportRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("LocalAITrainer", isDirectory: true)
    }

    static var runsDirectory: URL {
        appSupportRoot.appendingPathComponent("runs", isDirectory: true)
    }

    static var previewsDirectory: URL {
        appSupportRoot.appendingPathComponent("previews", isDirectory: true)
    }

    static func ensureBaseDirectories() throws {
        try FileManager.default.createDirectory(at: appSupportRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: runsDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: previewsDirectory, withIntermediateDirectories: true)
    }
}

enum WorkerLocator {
    static func scriptURL() throws -> URL {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        if let explicit = ProcessInfo.processInfo.environment["LOCAL_AI_TRAINER_WORKER"] {
            candidates.append(URL(fileURLWithPath: explicit))
        }

        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        candidates.append(currentDirectory.appendingPathComponent("worker/launch_worker.py"))
        candidates.append(currentDirectory.appendingPathComponent("../worker/launch_worker.py"))

        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0], isDirectory: false)
        var searchRoot = executableURL.deletingLastPathComponent()
        for _ in 0..<8 {
            candidates.append(searchRoot.appendingPathComponent("worker/launch_worker.py"))
            searchRoot.deleteLastPathComponent()
        }

        for candidate in candidates {
            if fileManager.fileExists(atPath: candidate.path) {
                return candidate.standardizedFileURL
            }
        }

        throw NSError(
            domain: "LocalAITrainer.WorkerLocator",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unable to locate worker/launch_worker.py from the current app context."]
        )
    }
}

enum BackendSetupLocator {
    static func scriptURL() throws -> URL {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        candidates.append(currentDirectory.appendingPathComponent("scripts/setup_worker_env.sh"))
        candidates.append(currentDirectory.appendingPathComponent("../scripts/setup_worker_env.sh"))

        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0], isDirectory: false)
        var searchRoot = executableURL.deletingLastPathComponent()
        for _ in 0..<8 {
            candidates.append(searchRoot.appendingPathComponent("scripts/setup_worker_env.sh"))
            searchRoot.deleteLastPathComponent()
        }

        for candidate in candidates where fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate.standardizedFileURL
        }

        throw NSError(
            domain: "LocalAITrainer.BackendSetupLocator",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Unable to locate scripts/setup_worker_env.sh from the current app context."]
        )
    }
}

enum PythonLocator {
    static func interpreterURL() -> URL {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        candidates.append(currentDirectory.appendingPathComponent(".venv/bin/python3"))
        candidates.append(currentDirectory.appendingPathComponent("../.venv/bin/python3"))

        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0], isDirectory: false)
        var searchRoot = executableURL.deletingLastPathComponent()
        for _ in 0..<8 {
            candidates.append(searchRoot.appendingPathComponent(".venv/bin/python3"))
            searchRoot.deleteLastPathComponent()
        }

        for candidate in candidates where fileManager.isExecutableFile(atPath: candidate.path) {
            return candidate.standardizedFileURL
        }

        return URL(fileURLWithPath: "/usr/bin/env")
    }

    static func bundledExecutableURL(named name: String) -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        candidates.append(currentDirectory.appendingPathComponent(".venv/bin/\(name)"))
        candidates.append(currentDirectory.appendingPathComponent("../.venv/bin/\(name)"))

        let executableURL = URL(fileURLWithPath: CommandLine.arguments[0], isDirectory: false)
        var searchRoot = executableURL.deletingLastPathComponent()
        for _ in 0..<8 {
            candidates.append(searchRoot.appendingPathComponent(".venv/bin/\(name)"))
            searchRoot.deleteLastPathComponent()
        }

        return candidates.first(where: { fileManager.isExecutableFile(atPath: $0.path) })?.standardizedFileURL
    }
}

enum OllamaLocator {
    private static let commonExecutablePaths = [
        "/opt/homebrew/bin/ollama",
        "/opt/homebrew/opt/ollama/bin/ollama",
        "/usr/local/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]

    static func executableURL() -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []

        if let explicit = ProcessInfo.processInfo.environment["LOCAL_AI_TRAINER_OLLAMA"] {
            candidates.append(URL(fileURLWithPath: explicit))
        }

        candidates.append(contentsOf: commonExecutablePaths.map { URL(fileURLWithPath: $0) })

        for directory in searchPaths() {
            candidates.append(URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent("ollama"))
        }

        return candidates.first(where: { fileManager.isExecutableFile(atPath: $0.path) })?.standardizedFileURL
    }

    static func searchPaths() -> [String] {
        var paths: [String] = []
        var seen = Set<String>()

        let environmentPaths = ProcessInfo.processInfo.environment["PATH"]?
            .split(separator: ":")
            .map(String.init) ?? []

        for path in environmentPaths + commonExecutablePaths.compactMap({ URL(fileURLWithPath: $0).deletingLastPathComponent().path }) {
            guard !path.isEmpty, !seen.contains(path) else { continue }
            seen.insert(path)
            paths.append(path)
        }

        return paths
    }
}

enum ProcessEnvironmentBuilder {
    static func defaultEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        let normalizedPath = OllamaLocator.searchPaths().joined(separator: ":")
        if !normalizedPath.isEmpty {
            environment["PATH"] = normalizedPath
        }
        if let executable = OllamaLocator.executableURL() {
            environment["LOCAL_AI_TRAINER_OLLAMA"] = executable.path
        }
        return environment
    }
}

final class ShellCommandRunner {
    func run(
        executable: String,
        arguments: [String],
        currentDirectory: URL? = nil,
        environment: [String: String]? = nil
    ) async throws -> ShellCommandResult {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            let outputPipe = Pipe()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.currentDirectoryURL = currentDirectory
            process.environment = environment ?? ProcessEnvironmentBuilder.defaultEnvironment()
            process.standardOutput = outputPipe
            process.standardError = outputPipe

            process.terminationHandler = { process in
                let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(decoding: data, as: UTF8.self)
                continuation.resume(returning: ShellCommandResult(exitCode: process.terminationStatus, output: output))
            }

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

enum HuggingFaceTokenStore {
    static func readToken() throws -> String? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        switch status {
        case errSecSuccess:
            guard
                let data = result as? Data,
                let token = String(data: data, encoding: .utf8),
                !token.isEmpty
            else {
                return nil
            }
            return token
        case errSecItemNotFound:
            return nil
        default:
            throw keychainError(status)
        }
    }

    static func saveToken(_ token: String) throws {
        let normalized = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else {
            throw NSError(
                domain: "LocalAITrainer.HuggingFaceTokenStore",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Enter a Hugging Face access token before saving."]
            )
        }

        let data = Data(normalized.utf8)
        let query = baseQuery()
        let attributes = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)

        if updateStatus == errSecItemNotFound {
            var newItem = query
            newItem[kSecValueData as String] = data
            newItem[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            let addStatus = SecItemAdd(newItem as CFDictionary, nil)
            guard addStatus == errSecSuccess else {
                throw keychainError(addStatus)
            }
            return
        }

        guard updateStatus == errSecSuccess else {
            throw keychainError(updateStatus)
        }
    }

    static func clearToken() throws {
        let status = SecItemDelete(baseQuery() as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw keychainError(status)
        }
    }

    private static func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.crissantiago.local-ai-trainer",
            kSecAttrAccount as String: "huggingface-access-token",
        ]
    }

    private static func keychainError(_ status: OSStatus) -> NSError {
        let message = SecCopyErrorMessageString(status, nil) as String? ?? "Keychain operation failed."
        return NSError(
            domain: "LocalAITrainer.HuggingFaceTokenStore",
            code: Int(status),
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

enum WorkerProcessEnvironment {
    static func current() -> [String: String] {
        var environment = ProcessEnvironmentBuilder.defaultEnvironment()
        if let token = try? HuggingFaceTokenStore.readToken(), !token.isEmpty {
            environment["HF_TOKEN"] = token
            environment["HUGGING_FACE_HUB_TOKEN"] = token
        }
        return environment
    }
}

final class EnvironmentInspector {
    private let commandRunner = ShellCommandRunner()
    private let ollamaService = OllamaService()
    private let preferredPython = PythonLocator.interpreterURL()

    func inspect() async -> EnvironmentStatus {
        var status = EnvironmentStatus()
        let hasBundledPython = FileManager.default.isExecutableFile(atPath: preferredPython.path)
        let hasBundledMflux = PythonLocator.bundledExecutableURL(named: "mflux-train") != nil
        status.physicalMemoryGB = Int((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0).rounded())
        status.availableDiskGB = Self.availableDiskGigabytes()

        #if arch(arm64)
        status.appleSilicon = true
        #else
        status.appleSilicon = false
        #endif

        status.fullXcodeInstalled = FileManager.default.fileExists(atPath: "/Applications/Xcode.app")

        status.ollamaInstalled = OllamaLocator.executableURL() != nil
        if hasBundledPython {
            status.pythonAvailable = true
        } else {
            status.pythonAvailable = await hasExecutable(named: "python3")
        }
        if hasBundledMflux {
            status.mfluxInstalled = true
        } else {
            status.mfluxInstalled = await hasExecutable(named: "mflux-train")
        }
        status.mlxLMInstalled = await hasPythonModule(named: "mlx_lm")

        if status.ollamaInstalled {
            status.ollamaDaemonReachable = (try? await ollamaService.isDaemonReachable()) ?? false
        }

        return status
    }

    private func hasExecutable(named command: String) async -> Bool {
        if command == "ollama" {
            return OllamaLocator.executableURL() != nil
        }
        guard let result = try? await commandRunner.run(executable: "/usr/bin/env", arguments: ["which", command]) else {
            return false
        }
        return result.exitCode == 0 && !result.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func hasPythonModule(named moduleName: String) async -> Bool {
        let script = "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('\(moduleName)') else 1)"
        let result: ShellCommandResult?
        if FileManager.default.isExecutableFile(atPath: preferredPython.path) {
            result = try? await commandRunner.run(executable: preferredPython.path, arguments: ["-c", script])
        } else {
            result = try? await commandRunner.run(executable: "/usr/bin/env", arguments: ["python3", "-c", script])
        }
        guard let result else {
            return false
        }
        return result.exitCode == 0
    }

    private static func availableDiskGigabytes() -> Int {
        let homeURL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        let values = try? homeURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let bytes = values?.volumeAvailableCapacityForImportantUsage ?? 0
        return Int((Double(bytes) / 1_073_741_824.0).rounded())
    }
}

final class OllamaService {
    func isDaemonReachable() async throws -> Bool {
        let url = URL(string: "http://127.0.0.1:11434/api/tags")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 3
        let (_, response) = try await URLSession.shared.data(for: request)
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    func listLocalModels() async throws -> [OllamaLocalModel] {
        let url = URL(string: "http://127.0.0.1:11434/api/tags")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let payload = try decoder.decode(TagsResponse.self, from: data)
        return payload.models
    }

    func pullModel(tag: String, onUpdate: @escaping @Sendable (OllamaPullProgress) async -> Void) async throws {
        let url = URL(string: "http://127.0.0.1:11434/api/pull")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 3_600
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(PullRequest(name: tag, stream: true))

        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let decoder = JSONDecoder()

        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            var errorBody = ""
            for try await line in bytes.lines {
                errorBody.append(line)
                errorBody.append("\n")
            }
            throw NSError(
                domain: "LocalAITrainer.Ollama",
                code: (response as? HTTPURLResponse)?.statusCode ?? 0,
                userInfo: [NSLocalizedDescriptionKey: errorBody.trimmingCharacters(in: .whitespacesAndNewlines)]
            )
        }

        for try await line in bytes.lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let payload = try decoder.decode(OllamaPullProgress.self, from: Data(trimmed.utf8))
            if let error = payload.error, !error.isEmpty {
                throw NSError(
                    domain: "LocalAITrainer.Ollama",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: error]
                )
            }
            await onUpdate(payload)
        }
    }

    func generateText(model: String, prompt: String) async throws -> String {
        let url = URL(string: "http://127.0.0.1:11434/api/generate")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = GenerateRequest(model: model, prompt: prompt, stream: false)
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await URLSession.shared.data(for: request)

        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw NSError(
                domain: "LocalAITrainer.Ollama",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: String(decoding: data, as: UTF8.self)]
            )
        }

        let decoder = JSONDecoder()
        let payload = try decoder.decode(GenerateResponse.self, from: data)
        return payload.response
    }

    private struct TagsResponse: Decodable {
        let models: [OllamaLocalModel]
    }

    private struct PullRequest: Encodable {
        let name: String
        let stream: Bool
    }

    private struct GenerateRequest: Encodable {
        let model: String
        let prompt: String
        let stream: Bool
    }

    private struct GenerateResponse: Decodable {
        let response: String
    }
}

final class TrainingAssetService {
    private let commandRunner = ShellCommandRunner()

    func inspectAssets(for model: SupportedModel) async throws -> TrainingAssetStatus {
        try await runAssetCommand(for: model, prepare: false)
    }

    func prepareAssets(for model: SupportedModel) async throws -> TrainingAssetStatus {
        let status = try await runAssetCommand(for: model, prepare: true)
        if !status.ready {
            throw NSError(
                domain: "LocalAITrainer.TrainingAssets",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: status.detail ?? "Training assets are not ready yet."]
            )
        }
        return status
    }

    private func runAssetCommand(for model: SupportedModel, prepare: Bool) async throws -> TrainingAssetStatus {
        let workerScript = try WorkerLocator.scriptURL()
        let pythonInterpreter = PythonLocator.interpreterURL()
        var arguments = [
            workerScript.path,
            "model-assets",
            "--workflow",
            model.workflowType.rawValue,
            "--backend-ref",
            model.trainingBackendRef,
        ]
        if prepare {
            arguments.append("--prepare")
        }

        let result = try await commandRunner.run(
            executable: pythonInterpreter.path,
            arguments: arguments,
            currentDirectory: workerScript.deletingLastPathComponent().deletingLastPathComponent(),
            environment: WorkerProcessEnvironment.current()
        )

        if result.exitCode != 0 {
            throw NSError(
                domain: "LocalAITrainer.TrainingAssets",
                code: Int(result.exitCode),
                userInfo: [NSLocalizedDescriptionKey: result.output.trimmingCharacters(in: .whitespacesAndNewlines)]
            )
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(TrainingAssetStatus.self, from: Data(result.output.utf8))
    }
}
