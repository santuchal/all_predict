import pandas as pd
from all_predict import LazyRegressorPlus
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

reg = LazyRegressorPlus(verbose=1)
models, predictions = reg.fit(X, y)

from all_predict.plotting import plot_performance

if 'R2' in models.columns:
    plot_performance(models, metric='R2')
else:
    print("'R2' metric not found in results. Available metrics:", models.columns.tolist())