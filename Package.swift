// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LocalAITrainer",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "LocalAITrainerApp", targets: ["LocalAITrainerApp"]),
    ],
    targets: [
        .executableTarget(
            name: "LocalAITrainerApp",
            path: "Sources/LocalAITrainerApp",
            exclude: ["Info.plist"],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/LocalAITrainerApp/Info.plist",
                ]),
            ]
        ),
    ]
)
