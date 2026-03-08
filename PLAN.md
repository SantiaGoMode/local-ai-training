# Local AI Trainer V1 Plan

## Product Goal

Ship a macOS desktop app that makes local fine-tuning approachable for no-code and low-code users on Apple Silicon, with a single UI for dataset selection, training, output inspection, and preview.

## What Changed In The Plan

The original concept assumed one Ollama-centered path for both text and image models. That is no longer the plan.

The current product plan is:

- FLUX image workflows:
  - train with MFLUX
  - keep the result as a LoRA adapter artifact
  - preview directly through MFLUX
  - do not require Ollama packaging
- Text workflows:
  - train from Hugging Face weights through MLX-LM
  - fuse the tuned result into a local model directory
  - import the fused model into Ollama for serving and comparison

This is the stable split that matches the backend tooling we validated.

## V1 User Experience.

- One main training page
- One settings page
- No terminal required for normal usage
- Automatic backend setup from inside the app
- Clear progress, warnings, and failure messages
- Output summary with saved artifact path and runtime target
- Before/after preview from the same prompt

## Supported V1 Model Strategy

- Image:
  - FLUX.2 klein via direct MFLUX runtime
- Text:
  - curated Ollama-compatible families only
  - Llama
  - Mistral
  - Gemma
  - Phi3

The curated list is intentional. V1 is not meant to support arbitrary local Ollama models.

## V1 Deliverables

- SwiftUI macOS shell
- Python worker for orchestration
- Environment and backend readiness checks
- Dataset validation and preprocessing
- Text fine-tuning path:
  - Hugging Face weights
  - MLX-LM LoRA fine-tuning
  - fused-model Ollama import
- Image fine-tuning path:
  - MFLUX LoRA training
  - direct adapter artifact output
  - direct before/after image preview
- Persisted job manifests and logs
- Signed and notarized DMG for private beta

## Success Criteria

- A non-technical user can install the app, select a supported model, select a dataset folder, and start training without using a terminal.
- A completed text run produces a usable local Ollama model.
- A completed FLUX image run produces a usable LoRA artifact and a working comparison preview.
- The app surfaces output paths, warnings, and failure reasons clearly enough that the user does not need backend logs to understand what happened.

## Remaining V1 Work

- Validate the updated text flow end-to-end in the app with current safer MLX presets.
- Validate an end-to-end FLUX train-plus-compare run through the direct MFLUX path.
- Validate one trained model in the downstream target application flow.
- Package the app as a signed, notarized `.dmg` with bundled Python dependencies.
- Add a clean release path using full Xcode instead of only the dev wrapper.

## Non-Goals For V1

- Packaging FLUX image adapters into new Ollama models
- Supporting arbitrary local Ollama model architectures
- Advanced hyperparameter editing in the main UI
- Cloud training

## Timeline

- Phase 1:
  - finalize current local workflows
  - complete text and FLUX validation
- Phase 2:
  - package the app for distribution
  - run clean-machine install validation
- Phase 3:
  - validate outputs in the downstream application
  - harden for private beta
