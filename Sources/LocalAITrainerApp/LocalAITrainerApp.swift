import SwiftUI

@main
struct LocalAITrainerApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup("Local AI Trainer") {
            RootView()
                .environmentObject(appState)
                .task {
                    await appState.bootstrap()
                }
                .frame(minWidth: 1080, minHeight: 780)
        }
    }
}

