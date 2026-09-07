import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


class SystemConfig(BaseModel):
    num_workers: int = Field(default=4, ge=1)
    batch_size: int = Field(default=50, ge=1)
    max_tasks_per_child: int = Field(default=500, ge=1)


class PathsConfig(BaseModel):
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    db_path: str = "data/tracker.db"
    log_dir: str = "logs"


class GeminiConfig(BaseModel):
    api_key: str = ""
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_retries: int = 8


class OCRConfig(BaseModel):
    engine: str = "gemini"
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)


class PreprocessingConfig(BaseModel):
    enable: bool = True
    max_dimension: int = 2560
    auto_orient: bool = True
    auto_contrast: bool = False
    split_double_pages: bool = False


class OutputConfig(BaseModel):
    format: str = "both"  # "txt", "jsonl", "both"
    save_confidence: bool = True
    mirror_folder_structure: bool = True


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path = "configs/config.yaml") -> "AppConfig":
        path = Path(yaml_path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
