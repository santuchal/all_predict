# __main__.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from all_predict import LazyRegressorPlus, LazyClassifierPlus

def main():
    parser = argparse.ArgumentParser(description="Run all_predict models on a dataset.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file containing data")
    parser.add_argument("--target", type=str, required=True, help="Name of the target column")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], required=True, help="Task type")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.file)
    X = df.drop(args.target, axis=1)
    y = df[args.target]

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.task == "regression":
        model_runner = LazyRegressorPlus(verbose=1)
    elif args.task == "regression":
    	model_runner = LazyClassifierPlus(verbose=1)
    else:
        print("Wrong argument!!!")
        exit(0)

    models, predictions = model_runner.fit(X,y)

    print(models)

if __name__ == "__main__":
    main()
