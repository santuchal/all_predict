"""Reporting and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def save_results(results: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save model comparison results to `model_results.csv`."""

    return save_dataframe(results, Path(output_dir) / "model_results.csv")


def save_predictions(predictions: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save predictions to `predictions.csv`."""

    return save_dataframe(predictions, Path(output_dir) / "predictions.csv")


def save_run_summary(metadata: dict[str, Any], output_dir: str | Path) -> Path:
    """Save run metadata to `run_summary.json`."""

    output_path = Path(output_dir) / "run_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
