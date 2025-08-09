# run_all_regressors_with_all_predict.py
# run_all_regressors_with_all_predict.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Import from the actual module path where LazyRegressorPlus is defined
from all_predict.regression import LazyRegressorPlus

# Load CSV generated earlier
df = pd.read_csv('svr_sample_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LazyRegressorPlus
reg = LazyRegressorPlus(verbose=1)

# Fit and get results
models, predictions = reg.fit(X,y)

print(models)

