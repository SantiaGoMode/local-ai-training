# Architecture

## Desktop Shell

- [Sources/LocalAITrainerApp](/Users/crissantiago/Documents/AI/Training/Sources/LocalAITrainerApp): SwiftUI macOS interface and app state
- The shell owns:
  - model selection
  - environment health
  - job launch and cancellation
  - progress display
  - output summaries
  - before/after preview UI

## Worker Layer

- [worker/local_ai_trainer_worker](/Users/crissantiago/Documents/AI/Training/worker/local_ai_trainer_worker): Python runtime for long-running backend work
- The worker owns:
  - dataset validation
  - dataset preprocessing
  - backend command execution
  - manifest updates
  - local artifact preparation
  - preview generation

The app and worker communicate through a JSON manifest contract. The worker updates status on disk while streaming logs back to the app.

## Runtime Split

### Text Workflow

- Input:
  - `.jsonl`
  - or `.txt` / `.md` folders that are preprocessed into train and validation JSONL files
- Base weights:
  - resolved from local Hugging Face snapshots
- Training:
  - MLX-LM LoRA fine-tuning
- Output:
  - fused local model directory
  - imported into Ollama for serving
- Preview:
  - before and after text generation through Ollama

### Image Workflow

- Input:
  - image folders with optional caption sidecars
- Base weights:
  - resolved from local MFLUX-compatible assets
- Training:
  - MFLUX LoRA fine-tuning
- Output:
  - direct LoRA adapter artifact
- Preview:
  - before and after image generation through direct MFLUX runtime

FLUX image runs do not become new Ollama models in the current architecture.

## Persistence

All local run state is written under:

```text
~/Library/Application Support/LocalAITrainer/runs/<job-id>/
```

That includes:

- manifests
- worker logs
- prepared datasets
- checkpoints
- fused text models
- image adapter artifacts

## Packaging Boundary

- The repository builds a development Swift target and launches through a dev `.app` wrapper.
- Shipping a signed `.app` and `.dmg` is a separate release step that requires full Xcode, signing credentials, notarization, and bundled Python dependencies.
