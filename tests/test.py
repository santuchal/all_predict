# from sklearn.datasets import make_regression
# from all_predict.regression import LazyRegressorPlus

# X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

# reg = LazyRegressorPlus(verbose=True)
# results, tuned = reg.fit(X, y)
# print(results.head())
# print(tuned)

# from sklearn.datasets import make_classification
# from all_predict.classification import LazyClassifierPlus
# import pandas as pd

# def save_classification_data_to_csv(n_samples, n_features, n_classes, random_state, filename="classification_data.csv"):
#     """
#     Generates synthetic classification data using make_classification and saves it to a CSV file.

#     Args:
#         n_samples (int): The number of samples to generate.
#         n_features (int): The number of features for each sample.
#         n_classes (int): The number of classes (labels).
#         random_state (int): The seed for random number generation for reproducibility.
#         filename (str): The name of the CSV file to save the data.
#     """
#     try:
#         # Generate the synthetic data using make_classification
#         # X will contain the features, and y will contain the labels
#         X, y = make_classification(
#             n_samples=n_samples,
#             n_features=n_features,
#             n_classes=n_classes,
#             random_state=random_state
#         )

#         # Create a DataFrame for the features.
#         # We'll name the columns 'feature_0', 'feature_1', etc.
#         feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
#         df_features = pd.DataFrame(X, columns=feature_columns)

#         # Add the target labels as a new column to the DataFrame
#         df_features['target'] = y

#         # Save the combined DataFrame to a CSV file.
#         # index=False ensures that pandas does not write the row index to the file.
#         df_features.to_csv(filename, index=False)

#         print(f"Successfully saved {n_samples} samples to '{filename}' with {n_features} features and {n_classes} classes.")
#         return X,y

#     except Exception as e:
#         print(f"An error occurred: {e}")

# # --- Example Usage ---
# # Use the user's requested parameters to demonstrate the function.
# save_classification_data_to_csv(
#     n_samples=500,
#     n_features=10,
#     n_classes=2,
#     random_state=42
# )


# X, y = save_classification_data_to_csv(n_samples=500, n_features=10, n_classes=2, random_state=42)
# # print(X)
# # print(y)
# # data = {
# #     X,y
# # }
# # df = pd.DataFrame(data)
# # df.to_csv('output_data.csv', index=False)
# print("Data saved successfully to 'output_data.csv'")


# clf = LazyClassifierPlus(verbose=True)
# results, tuned = clf.fit(X, y)
# print(results.head())
# print(tuned)
# clf.plot_results(results, metric_column='R2')

# from sklearn.datasets import make_regression
# from all_predict.regression import LazyRegressorPlus

# X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

# reg = LazyRegressorPlus(verbose=True)
# results, tuned = reg.fit(X, y)
# print(results.head())
# print(tuned)


import pandas as pd
from all_predict import LazyRegressorPlus

# Load your dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def plot_results(df, metric_column):
    """Plot model performance sorted by a given metric."""
    df_sorted = df.sort_values(by=metric_column, ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(df_sorted.index, df_sorted[metric_column], color='skyblue')
    plt.xlabel(metric_column)
    plt.ylabel('Models')
    plt.title(f'Model Comparison ({metric_column})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run all regressors
reg = LazyRegressorPlus(verbose=1)
models, predictions = reg.fit(X,y)
print(models.head())
print(predictions)
# Plot results
plot_results(models, metric_column='R2')