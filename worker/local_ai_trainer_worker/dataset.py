from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil

from .contracts import DatasetValidation, TextPreprocessResult


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}
JSONL_EXTENSION = ".jsonl"


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "untitled"


def validate_image_dataset(source_dir: str | Path) -> DatasetValidation:
    root = Path(source_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Image dataset directory does not exist: {root}")

    image_files = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not image_files:
        raise ValueError(f"No image files were found in {root}")

    warnings: list[str] = []
    captioned = 0
    missing_captions = 0
    for image_path in image_files:
        caption_path = image_path.with_suffix(".txt")
        if caption_path.exists():
            captioned += 1
        else:
            missing_captions += 1
            warnings.append(f"Missing caption sidecar for {image_path.name}; a draft prompt will be generated.")

    details = {
        "root": str(root),
        "image_count": len(image_files),
        "captioned_count": captioned,
        "missing_caption_count": missing_captions,
    }
    return DatasetValidation(workflow_type="image", sample_count=len(image_files), warnings=warnings, details=details)


def validate_text_dataset(source_dir: str | Path) -> DatasetValidation:
    root = Path(source_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Text dataset directory does not exist: {root}")

    jsonl_files = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == JSONL_EXTENSION]
    )
    text_files = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS]
    )
    if not jsonl_files and not text_files:
        raise ValueError(f"No supported text files (.txt, .md, .jsonl) were found in {root}")

    warnings: list[str] = []
    invalid_lines = 0
    jsonl_records = 0
    for jsonl_path in jsonl_files:
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {jsonl_path.name}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                invalid_lines += 1
                warnings.append(f"Non-object JSONL record in {jsonl_path.name}:{line_number}; it will be skipped.")
                continue
            jsonl_records += 1

    text_records = 0
    for text_path in text_files:
        contents = text_path.read_text(encoding="utf-8").strip()
        if contents:
            text_records += 1

    details = {
        "root": str(root),
        "jsonl_file_count": len(jsonl_files),
        "jsonl_record_count": jsonl_records,
        "text_file_count": len(text_files),
        "text_record_count": text_records,
        "invalid_jsonl_records": invalid_lines,
    }
    return DatasetValidation(
        workflow_type="text",
        sample_count=jsonl_records + text_records,
        warnings=warnings,
        details=details,
    )


def preprocess_text_dataset(source_dir: str | Path, output_dir: str | Path) -> TextPreprocessResult:
    root = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    warnings: list[str] = []

    for jsonl_path in sorted(root.rglob(f"*{JSONL_EXTENSION}")):
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                warnings.append(f"Skipped non-object JSONL record in {jsonl_path.name}:{line_number}.")
                continue
            records.append(payload)

    for text_path in sorted(root.rglob("*")):
        if not text_path.is_file() or text_path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        contents = text_path.read_text(encoding="utf-8").strip()
        if not contents:
            warnings.append(f"Skipped empty text file {text_path.name}.")
            continue
        records.append(
            {
                "text": contents,
                "metadata": {
                    "source_file": str(text_path.relative_to(root)),
                    "kind": text_path.suffix.lower().lstrip("."),
                },
            }
        )

    if not records:
        raise ValueError("No records were produced during text preprocessing.")

    train_records, valid_records = _split_records(records)
    train_path = destination / "train.jsonl"
    valid_path = destination / "valid.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(valid_path, valid_records)

    summary_path = destination / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "record_count": len(records),
                "train_count": len(train_records),
                "valid_count": len(valid_records),
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TextPreprocessResult(
        dataset_path=str(destination),
        record_count=len(records),
        warnings=warnings,
        details={"train_count": len(train_records), "valid_count": len(valid_records), "summary_path": str(summary_path)},
    )


def prepare_image_dataset(source_dir: str | Path, output_dir: str | Path, project_name: str) -> DatasetValidation:
    root = Path(source_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not image_files:
        raise ValueError(f"No image files were found in {root}")

    warnings: list[str] = []
    for index, image_path in enumerate(image_files, start=1):
        ext = image_path.suffix.lower()
        target_image = destination / f"sample-{index:04d}{ext}"
        shutil.copy2(image_path, target_image)

        source_caption = image_path.with_suffix(".txt")
        if source_caption.exists():
            caption = source_caption.read_text(encoding="utf-8").strip()
        else:
            warnings.append(f"Generated a caption sidecar for {image_path.name}.")
            caption = _draft_image_prompt(project_name=project_name, source_name=image_path.stem)
        (destination / f"sample-{index:04d}.txt").write_text(caption, encoding="utf-8")

        if index == 1:
            (destination / "preview1.txt").write_text(caption, encoding="utf-8")
    details = {
        "prepared_root": str(destination),
        "image_count": len(image_files),
        "preview_prompt": str(destination / "preview1.txt"),
    }
    return DatasetValidation(workflow_type="image", sample_count=len(image_files), warnings=warnings, details=details)


def _split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(records) == 1:
        return records, records

    valid_size = max(1, len(records) // 10)
    split_index = max(1, len(records) - valid_size)
    train_records = records[:split_index]
    valid_records = records[split_index:]
    return train_records, valid_records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _draft_image_prompt(project_name: str, source_name: str) -> str:
    readable_name = source_name.replace("-", " ").replace("_", " ").strip()
    return f"{project_name} product photography, {readable_name}, clean composition, marketing-ready imagery"
