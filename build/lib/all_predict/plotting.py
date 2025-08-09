# plotting.py
"""Plotting utilities for all_predict.
Generates visual comparisons of model performance and timing.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance(df_results, task="regression", metric=None, top_n=10):
    """Plots performance comparison of top_n models by a specified metric."""
    if metric is None:
        metric = "R2" if task == "regression" else "Accuracy"
    df_top = df_results.nlargest(top_n, metric)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metric, y="Model", data=df_top, palette="viridis")
    plt.title(f"Top {top_n} Models by {metric}")
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

def plot_timing(df_results, top_n=10):
    """Plots training and prediction time for top_n models."""
    df_top = df_results.head(top_n)
    plt.figure(figsize=(12, 6))
    df_melt = df_top.melt(id_vars=["Model"], value_vars=["Train Time", "Predict Time"], var_name="Type", value_name="Time (s)")
    sns.barplot(x="Time (s)", y="Model", hue="Type", data=df_melt, palette="magma")
    plt.title(f"Training and Prediction Time for Top {top_n} Models")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Model")
    plt.legend(title="Time Type")
    plt.tight_layout()
    plt.show()
