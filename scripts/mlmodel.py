#!/usr/bin/env python
# coding: utf-8

"""
Roller Coaster Cost Prediction Script (final version)
----------------------------------------------------
- Loads imputed dataset
- Trains Linear Regression, Random Forest, XGBoost
- Handles missing features via imputation pipeline
- Saves evaluation metrics to CSV
- Generates prediction and correlation plots
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Config ---
PLOTS_DIR = Path("./plots")
OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = ["Gforce", "Speed_mph", "Height_ft", "Inversions"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)


def load_data(path="./data/coaster_db_imputed.csv") -> pd.DataFrame:
    logging.info("Loading imputed dataset...")
    df = pd.read_csv(path)
    df = df[df["Cost_clean"].notna()]  # target should always exist
    return df


def correlation_heatmap(df: pd.DataFrame):
    logging.info("Plotting correlation heatmap...")
    plt.figure(figsize=(6, 4))
    corr = df[FEATURE_COLS + ["Cost_clean"]].corr()
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()


def train_models(X_train, y_train):
    logging.info("Training models...")

    pipelines = {
        "Linear Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.1,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    for name, pipe in pipelines.items():
        logging.info(f"Training {name}...")
        pipe.fit(X_train, y_train)
    return pipelines


def evaluate_models(models, X_test, y_test):
    logging.info("Evaluating models...")
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
            }
        )
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
    logging.info(f"Saved metrics to {OUTPUTS_DIR / 'model_metrics.csv'}")
    return df_metrics


def plot_predictions(models, X_test, y_test, y_full):
    logging.info("Plotting predictions...")
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 4))
    for i, (name, model) in enumerate(models.items(), 1):
        y_pred = model.predict(X_test)
        plt.subplot(1, 3, i)
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
        plt.plot(
            [y_full.min(), y_full.max()],
            [y_full.min(), y_full.max()],
            "r--",
            label="Perfect Fit",
        )
        plt.xlabel("Actual (log scale)")
        plt.ylabel("Predicted (log scale)")
        plt.title(name)
        plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predictions.png")
    plt.close()


def main():
    df = load_data()
    correlation_heatmap(df)

    X = df[FEATURE_COLS]
    y = np.log1p(df["Cost_clean"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    metrics_df = evaluate_models(models, X_test, y_test)
    logging.info(f"\nModel Comparison:\n{metrics_df}")

    plot_predictions(models, X_test, y_test, y_full=y)


if __name__ == "__main__":
    main()
