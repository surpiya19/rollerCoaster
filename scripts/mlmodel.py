#!/usr/bin/env python
# coding: utf-8

"""
Roller Coaster Cost Prediction Script
--------------------------------------
This script trains and evaluates ML models (Linear Regression, Random Forest,
and XGBoost) to predict coaster construction cost based on ride features.
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Config ---
PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)


def load_data(path="./data/coaster_db_clean.csv") -> pd.DataFrame:
    """Load and preprocess coaster dataset for modeling."""
    logging.info("Loading data...")
    df = pd.read_csv(path)
    return df[["Cost", "Gforce", "Speed_mph"]].dropna()


def correlation_heatmap(df: pd.DataFrame):
    """Plot and save correlation heatmap of features."""
    logging.info("Plotting correlation heatmap...")
    plt.figure(figsize=(6, 4))
    corr = df.corr()
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")

    # Add colorbar label
    from matplotlib.colorbar import Colorbar

    cbar: Colorbar = ax.collections[0].colorbar  # type: ignore[attr-defined]
    cbar.set_label("Correlation Strength", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()


def train_models(X_train, y_train):
    """Train multiple regression models."""
    logging.info("Training models...")
    models = {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random Forest": RandomForestRegressor(
            random_state=42, n_estimators=200, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test) -> pd.DataFrame:
    """Evaluate trained models on test data."""
    logging.info("Evaluating models...")
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(results)


def plot_predictions(models, X_test, y_test, y):
    """Scatter plot of actual vs predicted values for all models."""
    logging.info("Plotting predictions...")
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 4))
    for i, (name, model) in enumerate(models.items(), 1):
        y_pred = model.predict(X_test)
        plt.subplot(1, 3, i)
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect Fit")
        plt.xlabel("Actual (log scale)")
        plt.ylabel("Predicted (log scale)")
        plt.title(name)
        plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predictions.png")
    plt.close()


def xgboost_detailed(models, X_test, y_test):
    """Detailed evaluation and plot for XGBoost."""
    logging.info("Evaluating XGBoost in detail...")
    y_pred_log = models["XGBoost"].predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2 = r2_score(y_test, y_pred_log)

    logging.info(f"XGBoost (log scale) — MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")

    # Back-transform to actual scale
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse_actual = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2_actual = r2_score(y_test_actual, y_pred_actual)

    logging.info(
        f"XGBoost (actual scale) — MAE={mae_actual:,.0f}, "
        f"RMSE={rmse_actual:,.0f}, R²={r2_actual:.2f}"
    )

    # Scatter plot
    plt.figure(figsize=(7, 7))
    plt.style.use("seaborn-v0_8")
    plt.scatter(y_test, y_pred_log, alpha=0.6, s=60, edgecolor="k", linewidth=0.5)
    lims = [min(y_test.min(), y_pred_log.min()), max(y_test.max(), y_pred_log.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect Prediction")
    plt.xlabel("Actual (log scale)", fontsize=12)
    plt.ylabel("Predicted (log scale)", fontsize=12)
    plt.title("XGBoost: Predicted vs Actual (log scale)", fontsize=14, weight="bold")
    plt.text(
        0.05,
        0.95,
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.2f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgboost_predictions.png")
    plt.close()


def main():
    df_model = load_data()
    correlation_heatmap(df_model)

    X = df_model[["Gforce", "Speed_mph"]]
    y = np.log1p(df_model["Cost"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = train_models(X_train, y_train)
    results_df = evaluate_models(models, X_test, y_test)
    logging.info(f"\nModel Comparison:\n{results_df}")

    plot_predictions(models, X_test, y_test, y)
    xgboost_detailed(models, X_test, y_test)


if __name__ == "__main__":
    main()
