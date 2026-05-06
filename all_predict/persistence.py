"""Model persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a fitted model with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path) -> Any:
    """Load a model from disk."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    return joblib.load(input_path)
