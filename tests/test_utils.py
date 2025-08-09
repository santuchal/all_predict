# tests/test_utils.py
"""Unit tests for utils."""
import os
import pytest
from sklearn.linear_model import LinearRegression
from all_predict.utils import save_model, load_model

def test_save_and_load_model(tmp_path):
    model = LinearRegression()
    file_path = tmp_path / "model.pkl"
    save_model(model, file_path)
    loaded = load_model(file_path)
    assert isinstance(loaded, LinearRegression)

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")
