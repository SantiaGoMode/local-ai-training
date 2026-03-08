from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


class ManifestStatus:
    QUEUED = "queued"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    PACKAGING = "packaging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DatasetValidation:
    workflow_type: str
    sample_count: int
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextPreprocessResult:
    dataset_path: str
    record_count: int
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingJobManifest:
    job_id: str
    project_name: str
    workflow_type: str
    base_model_id: str
    base_model_tag: str
    training_backend_ref: str
    dataset_path: str
    output_dir: str
    preset: str
    packaging_strategy: str
    created_at: str
    status: str = ManifestStatus.QUEUED
    logs_path: str = ""
    derived_dataset_path: str | None = None
    output_artifact_path: str | None = None
    ollama_model_tag: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    preview_type: str | None = None
    preview_prompt: str | None = None
    preview_output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_path(cls, path: str | Path) -> "TrainingJobManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

