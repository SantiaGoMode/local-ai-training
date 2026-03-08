from __future__ import annotations

import argparse
import codecs
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

from .contracts import ManifestStatus, TrainingJobManifest
from .dataset import prepare_image_dataset, preprocess_text_dataset, validate_image_dataset, validate_text_dataset
from .ollama import build_custom_model_tag, create_model, prepare_fused_model_for_import, sanitize_cli_text, write_derived_modelfile


LOG_HANDLE = None
ACTIVE_MANIFEST: TrainingJobManifest | None = None
ACTIVE_MANIFEST_PATH: Path | None = None
ACTIVE_CHILD_PROCESS: subprocess.Popen[str] | None = None

TEXT_PRESET_CONFIG = {
    "fast": {
        "iters": 40,
        "batch_size": 1,
        "num_layers": 4,
        "learning_rate": 1e-5,
        "steps_per_report": 20,
        "steps_per_eval": 20,
        "save_every": 40,
        "max_seq_length": 1024,
        "val_batches": 2,
        "grad_checkpoint": False,
        "grad_accumulation_steps": 1,
    },
    "balanced": {
        "iters": 80,
        "batch_size": 1,
        "num_layers": 6,
        "learning_rate": 1e-5,
        "steps_per_report": 40,
        "steps_per_eval": 40,
        "save_every": 80,
        "max_seq_length": 1024,
        "val_batches": 2,
        "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
    },
    "quality": {
        "iters": 120,
        "batch_size": 1,
        "num_layers": 8,
        "learning_rate": 1e-5,
        "steps_per_report": 40,
        "steps_per_eval": 60,
        "save_every": 120,
        "max_seq_length": 1024,
        "val_batches": 2,
        "grad_checkpoint": True,
        "grad_accumulation_steps": 1,
    },
}

IMAGE_PRESET_CONFIG = {
    "fast": {"steps": 12, "epochs": 10, "learning_rate": 1e-4, "save_frequency": 5},
    "balanced": {"steps": 24, "epochs": 20, "learning_rate": 1e-4, "save_frequency": 10},
    "quality": {"steps": 40, "epochs": 40, "learning_rate": 7e-5, "save_frequency": 20},
}

FLUX2_LORA_TARGETS = [
    {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_k", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_v", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_out", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_q_proj", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_k_proj", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.add_v_proj", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.attn.to_add_out", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff.linear_in", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff.linear_out", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff_context.linear_in", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "transformer_blocks.{block}.ff_context.linear_out", "blocks": {"start": 0, "end": 5}, "rank": 16},
    {"module_path": "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", "blocks": {"start": 0, "end": 20}, "rank": 16},
    {"module_path": "single_transformer_blocks.{block}.attn.to_out", "blocks": {"start": 0, "end": 20}, "rank": 16},
]

FLUX2_TRAINING_REPOS = {
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
}

MLX_LM_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]

PROGRESS_SPINNER_CHARACTERS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏|/-\\")

GENERIC_TEXT_MEMORY_PROFILE = {
    "max_batch_size": 1,
    "max_num_layers": 8,
    "max_seq_length": 1024,
    "force_grad_checkpoint": True,
}

TEXT_MODEL_MEMORY_PROFILES = {
    "mistralai/Mistral-7B-Instruct-v0.3": GENERIC_TEXT_MEMORY_PROFILE,
    "meta-llama/Llama-3.1-8B-Instruct": GENERIC_TEXT_MEMORY_PROFILE,
    "google/gemma-2-9b-it": {
        "max_batch_size": 1,
        "max_num_layers": 12,
        "max_seq_length": 2048,
        "force_grad_checkpoint": True,
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "max_batch_size": 2,
        "max_num_layers": 16,
        "max_seq_length": 2048,
        "force_grad_checkpoint": False,
    },
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-ai-trainer-worker")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate-dataset", help="Validate a dataset folder.")
    validate_parser.add_argument("--workflow", choices=["image", "text"], required=True)
    validate_parser.add_argument("--input", required=True)
    validate_parser.set_defaults(handler=_handle_validate_dataset)

    preprocess_parser = subparsers.add_parser("preprocess-text", help="Preprocess text data into JSONL splits.")
    preprocess_parser.add_argument("--input", required=True)
    preprocess_parser.add_argument("--output", required=True)
    preprocess_parser.set_defaults(handler=_handle_preprocess_text)

    preview_parser = subparsers.add_parser("preview-image", help="Generate an image preview via MFLUX.")
    preview_parser.add_argument("--backend-ref", required=True)
    preview_parser.add_argument("--prompt", required=True)
    preview_parser.add_argument("--output-dir", required=True)
    preview_parser.add_argument("--model-path", required=False)
    preview_parser.add_argument("--lora-path", required=False)
    preview_parser.add_argument("--seed", type=int, default=42)
    preview_parser.add_argument("--steps", type=int, default=None)
    preview_parser.set_defaults(handler=_handle_preview_image)

    assets_parser = subparsers.add_parser("model-assets", help="Inspect or prepare managed training assets.")
    assets_parser.add_argument("--workflow", choices=["image", "text"], required=True)
    assets_parser.add_argument("--backend-ref", required=True)
    assets_parser.add_argument("--prepare", action="store_true")
    assets_parser.set_defaults(handler=_handle_model_assets)

    run_parser = subparsers.add_parser("run-job", help="Execute a training job from a manifest.")
    run_parser.add_argument("--manifest", required=True)
    run_parser.set_defaults(handler=_handle_run_job)

    return parser


def _handle_validate_dataset(args: argparse.Namespace) -> int:
    result = validate_image_dataset(args.input) if args.workflow == "image" else validate_text_dataset(args.input)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _handle_preprocess_text(args: argparse.Namespace) -> int:
    result = preprocess_text_dataset(args.input, args.output)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _handle_preview_image(args: argparse.Namespace) -> int:
    preview_output = io.StringIO()
    try:
        with redirect_stdout(preview_output), redirect_stderr(preview_output):
            path = _generate_flux_preview_image(
                backend_ref=args.backend_ref,
                prompt=args.prompt,
                output_dir=args.output_dir,
                model_path=args.model_path,
                lora_path=args.lora_path,
                seed=args.seed,
                steps=args.steps,
            )
    except Exception:  # noqa: BLE001
        diagnostic_output = preview_output.getvalue().strip()
        if diagnostic_output:
            print(diagnostic_output, file=sys.stderr)
        raise
    print(json.dumps({"image_path": str(path)}, indent=2))
    return 0


def _handle_model_assets(args: argparse.Namespace) -> int:
    payload = _managed_model_assets(
        workflow=args.workflow,
        backend_ref=args.backend_ref,
        prepare=args.prepare,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _handle_run_job(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = TrainingJobManifest.from_path(manifest_path)

    _install_signal_handlers()
    _activate_manifest(manifest, manifest_path)
    _log(f"Starting job {manifest.job_id} for workflow '{manifest.workflow_type}'.")

    try:
        if manifest.workflow_type == "text":
            _run_text_workflow(manifest)
        elif manifest.workflow_type == "image":
            _run_image_workflow(manifest)
        else:
            raise ValueError(f"Unsupported workflow type: {manifest.workflow_type}")
        manifest.status = ManifestStatus.COMPLETED
        _save_manifest()
        _log(f"Job {manifest.job_id} completed successfully.")
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.status = ManifestStatus.FAILED
        manifest.error_message = str(exc)
        _save_manifest()
        _log(f"Job {manifest.job_id} failed: {exc}")
        return 1
    finally:
        _close_log()


def _run_text_workflow(manifest: TrainingJobManifest) -> None:
    manifest.status = ManifestStatus.VALIDATING
    validation = validate_text_dataset(manifest.dataset_path)
    manifest.metadata["dataset_validation"] = validation.to_dict()
    manifest.warnings.extend(validation.warnings)
    _save_manifest()
    _log(f"Validated text dataset with {validation.sample_count} records.")

    manifest.status = ManifestStatus.PREPROCESSING
    derived_dir = Path(manifest.output_dir) / "derived_text_dataset"
    preprocess_result = preprocess_text_dataset(manifest.dataset_path, derived_dir)
    manifest.derived_dataset_path = preprocess_result.dataset_path
    manifest.metadata["text_preprocess"] = preprocess_result.to_dict()
    manifest.warnings.extend(preprocess_result.warnings)
    _save_manifest()
    _log(f"Prepared text dataset at {preprocess_result.dataset_path}.")

    train_count = int(preprocess_result.details.get("train_count", preprocess_result.record_count))
    valid_count = int(preprocess_result.details.get("valid_count", train_count))
    training_config, training_warnings = _resolve_text_training_config(
        preset=manifest.preset,
        train_count=train_count,
        valid_count=valid_count,
        training_backend_ref=manifest.training_backend_ref,
    )
    manifest.metadata["effective_text_training_config"] = training_config
    manifest.warnings.extend(training_warnings)
    _save_manifest()
    for warning in training_warnings:
        _log(warning)

    if not _has_python_module("mlx_lm"):
        raise RuntimeError("The Python worker environment is missing `mlx_lm`. Run ./scripts/setup_worker_env.sh first.")

    manifest.status = ManifestStatus.TRAINING
    adapter_dir = Path(manifest.output_dir) / "artifacts" / "mlx_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    command = _build_mlx_lm_command(
        training_backend_ref=manifest.training_backend_ref,
        dataset_dir=Path(preprocess_result.dataset_path),
        adapter_dir=adapter_dir,
        training_config=training_config,
        project_name=manifest.project_name,
        model_path=manifest.metadata.get("trainingModelPath"),
    )
    _log(f"Running MLX-LM fine-tuning command: {' '.join(command)}")
    _run_streaming_command(command, cwd=Path(manifest.output_dir))

    manifest.status = ManifestStatus.PACKAGING
    manifest.preview_type = "text"
    manifest.preview_prompt = "Create a concise marketing summary for fresh organic produce."
    model_tag = build_custom_model_tag(manifest.project_name, manifest.job_id)
    fused_model_dir = _fuse_text_model_for_ollama(
        base_model_tag=manifest.base_model_tag,
        training_model_path=manifest.metadata.get("trainingModelPath") or manifest.training_backend_ref,
        adapter_path=adapter_dir,
        output_dir=Path(manifest.output_dir),
        model_tag=model_tag,
    )

    manifest.output_artifact_path = str(fused_model_dir)
    manifest.metadata["adapterArtifactPath"] = str(adapter_dir)
    manifest.ollama_model_tag = model_tag
    _save_manifest()


def _run_image_workflow(manifest: TrainingJobManifest) -> None:
    manifest.status = ManifestStatus.VALIDATING
    validation = validate_image_dataset(manifest.dataset_path)
    manifest.metadata["dataset_validation"] = validation.to_dict()
    manifest.warnings.extend(validation.warnings)
    _save_manifest()
    _log(f"Validated image dataset with {validation.sample_count} images.")

    manifest.status = ManifestStatus.PREPROCESSING
    prepared_dir = Path(manifest.output_dir) / "prepared_image_dataset"
    prepared_result = prepare_image_dataset(manifest.dataset_path, prepared_dir, manifest.project_name)
    manifest.derived_dataset_path = str(prepared_dir)
    manifest.metadata["image_preparation"] = prepared_result.to_dict()
    manifest.warnings.extend(prepared_result.warnings)
    _save_manifest()
    _log(f"Prepared image dataset at {prepared_dir}.")

    mflux_executable = _mflux_train_executable()
    if not mflux_executable:
        raise RuntimeError("The Python worker environment is missing `mflux-train`. Run ./scripts/setup_worker_env.sh first.")

    manifest.status = ManifestStatus.TRAINING
    training_output_dir = Path(manifest.output_dir) / "mflux_training"
    training_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_mflux_config(
        destination=Path(manifest.output_dir) / "mflux_train.json",
        prepared_data_dir=prepared_dir,
        output_dir=training_output_dir,
        backend_ref=manifest.training_backend_ref,
        preset=manifest.preset,
        model_path=manifest.metadata.get("trainingModelPath"),
    )
    command = [mflux_executable, "--model", manifest.training_backend_ref, "--config", str(config_path)]
    _log(f"Running MFLUX training command: {' '.join(command)}")
    _run_streaming_command(command, cwd=Path(manifest.output_dir))

    manifest.status = ManifestStatus.PACKAGING
    manifest.preview_type = "image"
    manifest.preview_prompt = "Create a polished social media product image for fresh produce."
    adapter_path = _extract_latest_mflux_adapter(training_output_dir, Path(manifest.output_dir) / "artifacts" / "flux_adapter")
    manifest.output_artifact_path = str(adapter_path)
    manifest.metadata["preview_runtime"] = "mflux-direct-lora"
    manifest.metadata["preview_base_model"] = manifest.training_backend_ref
    manifest.warnings.append(
        "FLUX image runs stay in direct MFLUX LoRA format. Use the saved adapter artifact for comparison previews instead of expecting a new Ollama model tag."
    )
    _save_manifest()
    _log(f"Saved FLUX LoRA adapter artifact to {adapter_path}.")
    _log("Skipping Ollama packaging for FLUX image models. The app will preview this run directly with MFLUX.")


def _managed_model_assets(*, workflow: str, backend_ref: str, prepare: bool) -> dict[str, object]:
    if workflow == "image":
        return _managed_mflux_assets(backend_ref=backend_ref, prepare=prepare)

    if workflow == "text":
        return _managed_mlx_lm_assets(backend_ref=backend_ref, prepare=prepare)

    return {
        "workflow": workflow,
        "backend_ref": backend_ref,
        "managed": False,
        "ready": True,
        "requires_download": False,
        "source": "unmanaged",
        "detail": "This workflow currently resolves training weights on demand.",
    }


def _managed_mflux_assets(*, backend_ref: str, prepare: bool) -> dict[str, object]:
    if not _has_python_module("mflux"):
        return {
            "workflow": "image",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": False,
            "requires_download": False,
            "source": "missing-runtime",
            "detail": "MFLUX is not installed in the worker environment.",
        }

    repo_id = FLUX2_TRAINING_REPOS.get(backend_ref)
    if repo_id is None:
        return {
            "workflow": "image",
            "backend_ref": backend_ref,
            "managed": False,
            "ready": True,
            "requires_download": False,
            "source": "unmanaged",
            "detail": "No managed training-asset resolver is registered for this image backend yet.",
        }

    from mflux.models.common.resolution.path_resolution import PathResolution
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    patterns = Flux2KleinWeightDefinition.get_download_patterns()

    cached_path = PathResolution._find_complete_cached_snapshot(repo_id, patterns)
    if cached_path is not None:
        return {
            "workflow": "image",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": True,
            "requires_download": False,
            "source": "huggingface-cache",
            "repo_id": repo_id,
            "local_path": str(cached_path),
            "detail": "Training weights are already available locally.",
        }

    if not prepare:
        return {
            "workflow": "image",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": False,
            "requires_download": True,
            "source": "missing",
            "repo_id": repo_id,
            "local_path": None,
            "detail": "Training weights are not installed yet.",
        }

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
        resolved_path = PathResolution.resolve(repo_id, patterns)

    return {
        "workflow": "image",
        "backend_ref": backend_ref,
        "managed": True,
        "ready": True,
        "requires_download": False,
        "source": "downloaded",
        "repo_id": repo_id,
        "local_path": str(resolved_path) if resolved_path is not None else None,
        "detail": "Training weights were prepared and cached locally.",
    }


def _managed_mlx_lm_assets(*, backend_ref: str, prepare: bool) -> dict[str, object]:
    if not _has_python_module("mlx_lm"):
        return {
            "workflow": "text",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": False,
            "requires_download": False,
            "source": "missing-runtime",
            "detail": "MLX-LM is not installed in the worker environment.",
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {
            "workflow": "text",
            "backend_ref": backend_ref,
            "managed": False,
            "ready": False,
            "requires_download": False,
            "source": "missing-dependency",
            "detail": "huggingface_hub is not installed in the worker environment.",
        }

    try:
        cached_path = Path(snapshot_download(backend_ref, local_files_only=True))
    except Exception:  # noqa: BLE001
        cached_path = None

    if cached_path is not None:
        return {
            "workflow": "text",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": True,
            "requires_download": False,
            "source": "huggingface-cache",
            "repo_id": backend_ref,
            "local_path": str(cached_path),
            "detail": "Training weights are already available locally.",
        }

    if not prepare:
        return {
            "workflow": "text",
            "backend_ref": backend_ref,
            "managed": True,
            "ready": False,
            "requires_download": True,
            "source": "missing",
            "repo_id": backend_ref,
            "local_path": None,
            "detail": "Training weights are not installed yet.",
        }

    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
        resolved_path = Path(snapshot_download(backend_ref, allow_patterns=MLX_LM_ALLOW_PATTERNS))

    return {
        "workflow": "text",
        "backend_ref": backend_ref,
        "managed": True,
        "ready": True,
        "requires_download": False,
        "source": "downloaded",
        "repo_id": backend_ref,
        "local_path": str(resolved_path),
        "detail": "Training weights were prepared and cached locally.",
    }


def _build_mlx_lm_command(
    *,
    training_backend_ref: str,
    dataset_dir: Path,
    adapter_dir: Path,
    training_config: dict[str, int | float | bool],
    project_name: str,
    model_path: str | None = None,
) -> list[str]:
    command = _mlx_lm_command_prefix()
    command.extend(
        [
            "--model",
            model_path or training_backend_ref,
            "--train",
            "--data",
            str(dataset_dir),
            "--fine-tune-type",
            "lora",
            "--batch-size",
            str(training_config["batch_size"]),
            "--iters",
            str(training_config["iters"]),
            "--val-batches",
            str(training_config["val_batches"]),
            "--learning-rate",
            str(training_config["learning_rate"]),
            "--steps-per-report",
            str(training_config["steps_per_report"]),
            "--steps-per-eval",
            str(training_config["steps_per_eval"]),
            "--grad-accumulation-steps",
            str(training_config["grad_accumulation_steps"]),
            "--save-every",
            str(training_config["save_every"]),
            "--max-seq-length",
            str(training_config["max_seq_length"]),
            "--adapter-path",
            str(adapter_dir),
            "--project-name",
            project_name,
            "--num-layers",
            str(training_config["num_layers"]),
        ]
    )
    if bool(training_config.get("grad_checkpoint")):
        command.append("--grad-checkpoint")
    return command


def _resolve_text_training_config(
    *,
    preset: str,
    train_count: int,
    valid_count: int,
    training_backend_ref: str,
) -> tuple[dict[str, int | float | bool], list[str]]:
    config = dict(TEXT_PRESET_CONFIG.get(preset, TEXT_PRESET_CONFIG["balanced"]))
    warnings: list[str] = []

    physical_memory_gb = _physical_memory_gb()
    memory_profile = _text_memory_profile(training_backend_ref, physical_memory_gb)

    requested_batch_size = int(config["batch_size"])
    requested_num_layers = int(config["num_layers"])
    requested_max_seq_length = int(config["max_seq_length"])

    if memory_profile is not None:
        max_batch_size = int(memory_profile["max_batch_size"])
        max_num_layers = int(memory_profile["max_num_layers"])
        max_seq_length = int(memory_profile["max_seq_length"])

        if requested_batch_size > max_batch_size:
            config["batch_size"] = max_batch_size
            warnings.append(
                "Adjusted text training batch size from "
                f"{requested_batch_size} to {max_batch_size} to stay within the local MLX memory budget "
                f"for {training_backend_ref} on this machine."
            )

        if requested_num_layers < 0 or requested_num_layers > max_num_layers:
            config["num_layers"] = max_num_layers
            source_layers = "all layers" if requested_num_layers < 0 else str(requested_num_layers)
            warnings.append(
                "Adjusted trainable text layers from "
                f"{source_layers} to {max_num_layers} to stay within the local MLX memory budget "
                f"for {training_backend_ref} on this machine."
            )

        if requested_max_seq_length > max_seq_length:
            config["max_seq_length"] = max_seq_length
            warnings.append(
                "Adjusted text max sequence length from "
                f"{requested_max_seq_length} to {max_seq_length} to reduce MLX memory pressure "
                f"for {training_backend_ref} on this machine."
            )

        if bool(memory_profile.get("force_grad_checkpoint")) and not bool(config.get("grad_checkpoint")):
            config["grad_checkpoint"] = True
            warnings.append(
                f"Enabled gradient checkpointing automatically to reduce MLX memory use for {training_backend_ref}."
            )

    if int(config["batch_size"]) < requested_batch_size and int(config.get("grad_accumulation_steps", 1)) == 1:
        preserved_effective_batch = max(1, requested_batch_size // int(config["batch_size"]))
        if preserved_effective_batch > 1:
            config["grad_accumulation_steps"] = preserved_effective_batch
            warnings.append(
                "Increased gradient accumulation steps to "
                f"{preserved_effective_batch} so the smaller memory-safe batch size keeps a similar effective update size."
            )

    candidate_sizes = [count for count in (train_count, valid_count) if count > 0]
    smallest_split = min(candidate_sizes) if candidate_sizes else 1
    effective_batch_size = max(1, min(int(config["batch_size"]), smallest_split))

    if effective_batch_size != int(config["batch_size"]):
        warnings.append(
            "Adjusted text training batch size from "
            f"{config['batch_size']} to {effective_batch_size} because the prepared dataset "
            f"split sizes are train={train_count}, valid={valid_count}."
        )
        config["batch_size"] = effective_batch_size

    max_val_batches = max(1, valid_count // max(1, int(config["batch_size"]))) if valid_count > 0 else 1
    effective_val_batches = max(1, min(int(config["val_batches"]), max_val_batches))
    if effective_val_batches != int(config["val_batches"]):
        warnings.append(
            "Adjusted validation batches from "
            f"{config['val_batches']} to {effective_val_batches} because the prepared validation split "
            f"supports at most {max_val_batches} batches at batch size {config['batch_size']}."
        )
        config["val_batches"] = effective_val_batches

    return config, warnings


def _text_memory_profile(training_backend_ref: str, physical_memory_gb: int) -> dict[str, int | bool] | None:
    profile = TEXT_MODEL_MEMORY_PROFILES.get(training_backend_ref)
    if profile is None:
        return None
    if physical_memory_gb > 36:
        return None
    return profile


def _physical_memory_gb() -> int:
    try:
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if page_count > 0 and page_size > 0:
            return max(1, round((page_count * page_size) / 1_073_741_824))
    except (AttributeError, OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            check=True,
            text=True,
        )
        return max(1, round(int(result.stdout.strip()) / 1_073_741_824))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 36


def _fuse_text_model_for_ollama(
    *,
    base_model_tag: str,
    training_model_path: str,
    adapter_path: Path,
    output_dir: Path,
    model_tag: str,
) -> Path:
    fused_model_dir = output_dir / "ollama" / "fused_model"
    command = _mlx_lm_fuse_command_prefix()
    command.extend(
        [
            "--model",
            training_model_path,
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(fused_model_dir),
        ]
    )
    _log(f"Fusing MLX adapter into a full model directory: {' '.join(command)}")
    _run_streaming_command(command, cwd=output_dir)

    with tempfile.TemporaryDirectory(prefix="local-ai-ollama-import-") as temp_dir:
        temp_root = Path(temp_dir)
        tokenizer_assets_dir = Path(training_model_path).expanduser().resolve() if Path(training_model_path).exists() else None
        import_model_dir = prepare_fused_model_for_import(
            fused_model_dir,
            temp_root / "model",
            tokenizer_assets_dir=tokenizer_assets_dir,
        )
        modelfile = write_derived_modelfile(
            base_model_tag=base_model_tag,
            imported_model_path=import_model_dir,
            destination=temp_root / "Modelfile",
            cwd=output_dir,
        )
        result = create_model(model_tag, modelfile, cwd=temp_root)
        combined_output = sanitize_cli_text(result.stdout) or sanitize_cli_text(result.stderr)
        if combined_output:
            _log(combined_output)
        if result.returncode != 0:
            raise RuntimeError(
                sanitize_cli_text(result.stderr)
                or sanitize_cli_text(result.stdout)
                or "Ollama model creation failed for the text workflow."
            )

    return fused_model_dir


def _write_mflux_config(
    *,
    destination: Path,
    prepared_data_dir: Path,
    output_dir: Path,
    backend_ref: str,
    preset: str,
    model_path: str | None = None,
) -> Path:
    config = IMAGE_PRESET_CONFIG.get(preset, IMAGE_PRESET_CONFIG["balanced"])
    timestep_low, timestep_high = _flux2_timestep_bounds(config["steps"])
    payload = {
        "model": backend_ref,
        "data": str(prepared_data_dir),
        "seed": 4,
        "steps": config["steps"],
        "guidance": 1.0,
        "quantize": None,
        "max_resolution": 1024,
        "low_ram": False,
        "training_loop": {
            "num_epochs": config["epochs"],
            "batch_size": 1,
            "timestep_low": timestep_low,
            "timestep_high": timestep_high,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config["learning_rate"],
        },
        "checkpoint": {
            "save_frequency": config["save_frequency"],
            "output_path": str(output_dir),
        },
        "monitoring": {
            "preview_width": 1024,
            "preview_height": 1024,
            "plot_frequency": 1,
            "generate_image_frequency": config["save_frequency"],
        },
        "lora_layers": {
            "targets": FLUX2_LORA_TARGETS,
        },
    }
    if model_path:
        payload["model_path"] = model_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _flux2_timestep_bounds(steps: int) -> tuple[int, int]:
    timestep_high = max(9, int(steps))
    timestep_low = round(timestep_high * 0.625)
    timestep_low = max(4, min(timestep_low, timestep_high - 1))
    return timestep_low, timestep_high


def _extract_latest_mflux_adapter(training_output_dir: Path, extraction_dir: Path) -> Path:
    checkpoint_files: list[Path] = []
    for candidate_dir in _candidate_mflux_output_dirs(training_output_dir):
        checkpoint_files.extend(candidate_dir.joinpath("checkpoints").glob("*_checkpoint.zip"))
    if not checkpoint_files:
        raise RuntimeError(
            f"No MFLUX checkpoint zip was found under {training_output_dir} or its timestamped output folders."
        )

    latest_checkpoint = max(checkpoint_files, key=lambda path: path.stat().st_mtime)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(latest_checkpoint, "r") as archive:
        adapter_members = [name for name in archive.namelist() if name.endswith("_adapter.safetensors")]
        if not adapter_members:
            raise RuntimeError(f"No LoRA adapter was found inside {latest_checkpoint.name}.")
        adapter_name = adapter_members[-1]
        archive.extract(adapter_name, path=extraction_dir)
        return extraction_dir / adapter_name


def _candidate_mflux_output_dirs(training_output_dir: Path) -> list[Path]:
    candidates = [training_output_dir]
    parent_dir = training_output_dir.parent
    if parent_dir.exists():
        candidates.extend(
            sorted(
                [
                    path
                    for path in parent_dir.glob(f"{training_output_dir.name}*")
                    if path.is_dir()
                ]
            )
        )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
    return unique_candidates


def _generate_flux_preview_image(
    *,
    backend_ref: str,
    prompt: str,
    output_dir: str | Path,
    model_path: str | None,
    lora_path: str | None,
    seed: int,
    steps: int | None,
) -> Path:
    if not _has_python_module("mflux"):
        raise RuntimeError("The Python worker environment is missing `mflux`. Run ./scripts/setup_worker_env.sh first.")

    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig.from_name(model_name=backend_ref)
    inference_steps = steps if steps is not None else (20 if "base" in backend_ref else 4)
    lora_paths = [str(Path(lora_path).expanduser().resolve())] if lora_path else None
    lora_scales = [1.0] if lora_paths else None

    model = Flux2Klein(
        model_config=model_config,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
    )

    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        width=1024,
        height=1024,
        guidance=1.0,
        num_inference_steps=inference_steps,
        scheduler="flow_match_euler_discrete",
    )

    suffix = f"{int(time.time())}-{seed}"
    output_path = target_dir / f"preview-{suffix}.png"
    image.save(output_path, overwrite=True)
    return output_path


def _mlx_lm_command_prefix() -> list[str]:
    sibling = Path(sys.executable).parent / "mlx_lm"
    if sibling.exists() and sibling.is_file():
        return [str(sibling), "lora"]

    entrypoint = shutil.which("mlx_lm")
    if entrypoint:
        return [entrypoint, "lora"]

    return [sys.executable, "-m", "mlx_lm", "lora"]


def _mlx_lm_fuse_command_prefix() -> list[str]:
    sibling = Path(sys.executable).parent / "mlx_lm.fuse"
    if sibling.exists() and sibling.is_file():
        return [str(sibling)]

    entrypoint = shutil.which("mlx_lm.fuse")
    if entrypoint:
        return [entrypoint]

    fallback = Path(sys.executable).parent / "mlx_lm"
    if fallback.exists() and fallback.is_file():
        return [str(fallback), "fuse"]

    entrypoint = shutil.which("mlx_lm")
    if entrypoint:
        return [entrypoint, "fuse"]

    return [sys.executable, "-m", "mlx_lm", "fuse"]


def _mflux_train_executable() -> str | None:
    sibling = Path(sys.executable).parent / "mflux-train"
    if sibling.exists() and sibling.is_file():
        return str(sibling)
    return shutil.which("mflux-train")


def _has_python_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _run_streaming_command(command: list[str], cwd: Path | None = None) -> None:
    global ACTIVE_CHILD_PROCESS
    captured_output: list[str] = []
    stream_state = {"buffer": "", "last_progress_message": ""}
    ACTIVE_CHILD_PROCESS = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )

    assert ACTIVE_CHILD_PROCESS.stdout is not None
    stream_reader = ACTIVE_CHILD_PROCESS.stdout
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        while True:
            if hasattr(stream_reader, "read1"):
                chunk = stream_reader.read1(4096)
            else:
                chunk = stream_reader.read(4096)
            if not chunk:
                break
            decoded_chunk = decoder.decode(chunk)
            _consume_stream_output(decoded_chunk, captured_output, stream_state)
        trailing_text = decoder.decode(b"", final=True)
        if trailing_text:
            _consume_stream_output(trailing_text, captured_output, stream_state)
        _flush_stream_buffer(captured_output, stream_state, is_progress=False)
    finally:
        stream_reader.close()

    exit_code = ACTIVE_CHILD_PROCESS.wait()
    ACTIVE_CHILD_PROCESS = None
    if exit_code != 0:
        failure_detail = _summarize_command_failure(captured_output)
        if failure_detail:
            raise RuntimeError(f"{failure_detail}\nCommand failed with exit code {exit_code}: {' '.join(command)}")
        raise RuntimeError(f"Command failed with exit code {exit_code}: {' '.join(command)}")


def _summarize_command_failure(captured_output: list[str]) -> str:
    interesting_markers = (
        "insufficient memory",
        "out of memory",
        "runtime_error",
        "traceback",
        "error:",
        "valueerror",
        "typeerror",
    )

    for line in reversed(captured_output):
        lowered = line.lower()
        if any(marker in lowered for marker in interesting_markers):
            return line

    for line in reversed(captured_output):
        if line:
            return line

    return ""


def _consume_stream_output(
    text: str,
    captured_output: list[str],
    stream_state: dict[str, str],
) -> None:
    for character in text:
        if character == "\r":
            _flush_stream_buffer(captured_output, stream_state, is_progress=True)
        elif character == "\n":
            _flush_stream_buffer(captured_output, stream_state, is_progress=False)
        else:
            stream_state["buffer"] += character


def _flush_stream_buffer(
    captured_output: list[str],
    stream_state: dict[str, str],
    *,
    is_progress: bool,
) -> None:
    raw_text = stream_state["buffer"]
    stream_state["buffer"] = ""

    sanitized = sanitize_cli_text(raw_text).strip()
    if not sanitized:
        return

    normalized_progress_message = _normalize_progress_message(sanitized)
    message = normalized_progress_message if is_progress or _looks_like_progress_message(sanitized) else sanitized
    if not message:
        return

    if is_progress:
        if message == stream_state["last_progress_message"]:
            return
        stream_state["last_progress_message"] = message
    else:
        stream_state["last_progress_message"] = ""

    captured_output.append(message)
    _log(message)


def _normalize_progress_message(message: str) -> str:
    normalized = " ".join(message.split())
    normalized = normalized.rstrip()
    while normalized and normalized[-1] in PROGRESS_SPINNER_CHARACTERS:
        normalized = normalized[:-1].rstrip()
    normalized = re.sub(r"\s+\d+\.\d+/s$", "", normalized)
    return normalized.strip()


def _looks_like_progress_message(message: str) -> bool:
    if any(character in PROGRESS_SPINNER_CHARACTERS for character in message):
        return True
    lowered = message.lower()
    return lowered.startswith("gathering model components") or lowered.startswith("copying file sha256:")


def _activate_manifest(manifest: TrainingJobManifest, manifest_path: Path) -> None:
    global ACTIVE_MANIFEST, ACTIVE_MANIFEST_PATH, LOG_HANDLE
    ACTIVE_MANIFEST = manifest
    ACTIVE_MANIFEST_PATH = manifest_path

    log_path = Path(manifest.logs_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_HANDLE = log_path.open("a", encoding="utf-8")
    _save_manifest()


def _save_manifest() -> None:
    if ACTIVE_MANIFEST is None or ACTIVE_MANIFEST_PATH is None:
        return
    ACTIVE_MANIFEST.write(ACTIVE_MANIFEST_PATH)


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if LOG_HANDLE is not None:
        LOG_HANDLE.write(line + "\n")
        LOG_HANDLE.flush()


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    signal.signal(signal.SIGINT, _handle_termination_signal)


def _handle_termination_signal(signum: int, _frame) -> None:
    del signum  # unused, but preserved for signal signature compatibility
    if ACTIVE_CHILD_PROCESS is not None and ACTIVE_CHILD_PROCESS.poll() is None:
        ACTIVE_CHILD_PROCESS.terminate()
    if ACTIVE_MANIFEST is not None:
        ACTIVE_MANIFEST.status = ManifestStatus.CANCELLED
        ACTIVE_MANIFEST.error_message = "Training was cancelled."
        _save_manifest()
    _log("Training job cancelled.")
    _close_log()
    raise SystemExit(130)


def _close_log() -> None:
    global LOG_HANDLE
    if LOG_HANDLE is not None:
        LOG_HANDLE.close()
        LOG_HANDLE = None
