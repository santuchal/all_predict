import subprocess
import sys

from sklearn.datasets import load_breast_cancer, load_diabetes


def test_cli_classification(tmp_path):
    data = load_breast_cancer(as_frame=True)
    frame = data.frame.copy()
    frame["target"] = data.target
    csv_path = tmp_path / "classification.csv"
    frame.to_csv(csv_path, index=False)

    output_dir = tmp_path / "classification_output"
    command = [
        sys.executable,
        "-m",
        "all_predict.cli",
        "classify",
        "--file",
        str(csv_path),
        "--target",
        "target",
        "--output",
        str(output_dir),
        "--include-models",
        "LogisticRegression,RandomForestClassifier,DummyClassifier",
        "--predictions",
        "--save-best",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (output_dir / "model_results.csv").exists()
    assert (output_dir / "run_summary.json").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "best_model.joblib").exists()


def test_cli_regression(tmp_path):
    data = load_diabetes(as_frame=True)
    frame = data.frame.copy()
    frame["target"] = data.target
    csv_path = tmp_path / "regression.csv"
    frame.to_csv(csv_path, index=False)

    output_dir = tmp_path / "regression_output"
    command = [
        sys.executable,
        "-m",
        "all_predict.cli",
        "regress",
        "--file",
        str(csv_path),
        "--target",
        "target",
        "--output",
        str(output_dir),
        "--include-models",
        "LinearRegression,RandomForestRegressor,DummyRegressor",
        "--predictions",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (output_dir / "model_results.csv").exists()
    assert (output_dir / "run_summary.json").exists()
