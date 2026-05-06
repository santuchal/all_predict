import json

import pandas as pd

from all_predict.reporting import save_predictions, save_results, save_run_summary


def test_reporting_files(tmp_path):
    results = pd.DataFrame(
        [{"Model": "Example", "Accuracy": 1.0, "Status": "success", "Error": ""}]
    )
    predictions = pd.DataFrame({"Example": [0, 1, 1]})
    summary = {"task": "classification", "best_model_name": "Example"}

    results_path = save_results(results, tmp_path)
    predictions_path = save_predictions(predictions, tmp_path)
    summary_path = save_run_summary(summary, tmp_path)

    assert results_path.exists()
    assert predictions_path.exists()
    assert summary_path.exists()
    assert pd.read_csv(results_path).iloc[0]["Model"] == "Example"
    assert json.loads(summary_path.read_text())["best_model_name"] == "Example"
