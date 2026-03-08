from __future__ import annotations

import os
from pathlib import Path
import json
import re
import shutil
import subprocess
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TOKENIZER_ASSET_FILES = (
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
)
COMMON_OLLAMA_PATHS = (
    "/opt/homebrew/bin/ollama",
    "/opt/homebrew/opt/ollama/bin/ollama",
    "/usr/local/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)


def ollama_path() -> str | None:
    explicit = os.environ.get("LOCAL_AI_TRAINER_OLLAMA")
    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)

    for candidate in COMMON_OLLAMA_PATHS:
        resolved = Path(candidate).expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)

    return shutil.which("ollama")


def ensure_ollama_installed() -> str:
    executable = ollama_path()
    if executable is None:
        raise RuntimeError("Ollama is not installed or is not on PATH.")
    return executable


def build_custom_model_tag(project_name: str, job_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-") or "project"
    tag_suffix = re.sub(r"[^a-z0-9]+", "-", job_id.lower()).strip("-")
    return f"custom/{slug}:{tag_suffix}"


def sanitize_cli_text(text: str) -> str:
    stripped = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n").strip()
    return stripped


def write_modelfile(base_model_tag: str, adapter_path: str | Path, destination: str | Path) -> Path:
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"FROM {base_model_tag}\nADAPTER {Path(adapter_path).expanduser().resolve()}\n",
        encoding="utf-8",
    )
    return target


def create_model(model_tag: str, modelfile_path: str | Path, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    executable = ensure_ollama_installed()
    return subprocess.run(
        [executable, "create", model_tag, "-f", str(Path(modelfile_path).expanduser().resolve())],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )


def show_modelfile(model_tag: str, cwd: str | Path | None = None) -> str:
    executable = ensure_ollama_installed()
    result = subprocess.run(
        [executable, "show", model_tag, "--modelfile"],
        cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            sanitize_cli_text(result.stdout)
            or sanitize_cli_text(result.stderr)
            or f"Unable to inspect the Ollama Modelfile for {model_tag}."
        )
    return sanitize_cli_text(result.stdout)


def write_derived_modelfile(
    *,
    base_model_tag: str,
    imported_model_path: str | Path,
    destination: str | Path,
    cwd: str | Path | None = None,
) -> Path:
    imported_model_path = Path(imported_model_path).expanduser().resolve()
    base_modelfile = show_modelfile(base_model_tag, cwd=cwd)

    body_lines = []
    replaced_from = False
    for line in base_modelfile.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("FROM ") and not replaced_from:
            body_lines.append(f"FROM {imported_model_path}")
            replaced_from = True
        else:
            body_lines.append(line)

    if not replaced_from:
        body_lines.insert(0, f"FROM {imported_model_path}")

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(body_lines).strip() + "\n", encoding="utf-8")
    return target


def prepare_fused_model_for_import(
    source_model_dir: str | Path,
    destination: str | Path,
    *,
    tokenizer_assets_dir: str | Path | None = None,
) -> Path:
    source_model_dir = Path(source_model_dir).expanduser().resolve()
    target = Path(destination).expanduser().resolve()

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source_model_dir, target, symlinks=False)

    config_path = target / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = config.get("model_type")
        if model_type == "mistral":
            config["architectures"] = ["LlamaForCausalLM"]
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if tokenizer_assets_dir is not None:
        _copy_missing_tokenizer_assets(Path(tokenizer_assets_dir).expanduser().resolve(), target)

    return target


def _copy_missing_tokenizer_assets(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return

    for name in TOKENIZER_ASSET_FILES:
        source_path = source_dir / name
        target_path = target_dir / name
        if source_path.exists() and not target_path.exists():
            shutil.copy2(source_path, target_path)


def generate_image(model_tag: str, prompt: str, output_dir: str | Path) -> Path:
    executable = ensure_ollama_installed()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    before = {path.name for path in _iter_image_files(target_dir)}
    result = subprocess.run(
        [executable, "run", model_tag, prompt],
        cwd=str(target_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            sanitize_cli_text(result.stdout)
            or sanitize_cli_text(result.stderr)
            or "Image preview generation failed."
        )

    created_files = [path for path in _iter_image_files(target_dir) if path.name not in before]
    if not created_files:
        raise RuntimeError("Ollama completed without writing a preview image to disk.")
    return max(created_files, key=lambda path: path.stat().st_mtime)


def _iter_image_files(directory: Path) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
