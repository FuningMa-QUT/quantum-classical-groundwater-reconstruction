"""Configuration loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file.

    JSON is dependency-free and preferred for the first upgraded version. YAML is
    supported when PyYAML is available in the active conda environment.
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Use .json configs or install PyYAML."
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")
        return loaded

    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def ensure_output_dir(root: str | Path, experiment_name: str) -> Path:
    out_dir = Path(root) / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

