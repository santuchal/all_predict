# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="all_predict",
    version="0.1.0",
    author="Santu Chall",
    author_email="santuchal@gmail.com",
    description="An enhanced LazyPredict-like library with more models, hyperparameter tuning, and visualizations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santuchal/all_predict",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0",
        "pandas>=1.3",
        "numpy>=1.21",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "joblib>=1.0",
        "xgboost>=1.5; python_version>='3.7'",
        "lightgbm>=3.3; python_version>='3.7'",
        "catboost>=1.0; python_version>='3.7'"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)