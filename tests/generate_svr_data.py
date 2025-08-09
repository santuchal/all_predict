# generate_svr_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic regression data
X, y = make_regression(n_samples=200, n_features=5, noise=0.2, random_state=42)

# Create a DataFrame
columns = [f"feature_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['target'] = y

# Save to CSV
df.to_csv('svr_sample_data.csv', index=False)
print("CSV file 'svr_sample_data.csv' created successfully.")