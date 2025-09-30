#!/usr/bin/env python3
# coding: utf-8

"""
Modeling Report: Roller Coaster Cost Prediction
------------------------------------------------
- Compares model performance before and after cost imputation
- Generates evaluation metrics and visualizations
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Config ---
RAW_DATA = Path("./data/coaster_db_clean.csv")
IMPUTED_DATA = Path("./data/coaster_db_imputed.csv")
OUTPUTS_DIR = Path("./outputs")
PLOTS_DIR = Path("./plots")
OUTPUTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
FEATURE_COLS = ["Gforce", "Speed_mph", "Height_ft", "Inversions"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)


def load_raw_data(path=RAW_DATA):
    logging.info("Loading raw dataset...")
    df = pd.read_csv(path)

    # Convert raw cost to numeric
    df["Cost"] = (
        df["Cost"]
        .astype(str)
        .str.replace(r"[^\d\.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    df = df.dropna(subset=FEATURE_COLS + ["Cost"])
    return df


def load_imputed_data(path=IMPUTED_DATA):
    logging.info("Loading imputed dataset...")
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURE_COLS + ["Cost_clean"])
    return df


def train_models(X_train, y_train):
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
                    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
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
    return pd.DataFrame(results)


def plot_before_after_comparison(raw_metrics, imputed_metrics):
    """
    Creates a bar plot comparing model performance before and after imputation.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Add dataset labels
    raw_metrics = raw_metrics.copy()
    raw_metrics["Dataset"] = "Raw"
    imputed_metrics = imputed_metrics.copy()
    imputed_metrics["Dataset"] = "Imputed"

    # Combine
    metrics_long = pd.concat([raw_metrics, imputed_metrics], ignore_index=True)
    metrics_long = metrics_long.melt(
        id_vars=["Model", "Dataset"],
        value_vars=["MAE", "RMSE", "R2"],
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=metrics_long,
        x="Metric",
        y="Value",
        hue="Dataset",
        ci=None,
        palette=["#FF9999", "#66B2FF"],
    )

    # Annotate values on bars
    for p in ax.patches:
        height = p.get_height()  # type: ignore
        if not np.isnan(height):
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),  # type: ignore
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )

    plt.title("Model Performance: Before vs After Imputation")
    plt.ylabel("Metric Value")
    plt.xlabel("Metric")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "before_after_imputation_comparison.png")
    plt.close()


def main():
    # --- Raw dataset ---
    raw_df = load_raw_data()
    X_raw = raw_df[FEATURE_COLS]
    y_raw = np.log1p(raw_df["Cost"])  # <- use "Cost" not "Cost_clean"

    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    raw_models = train_models(X_tr_raw, y_tr_raw)
    raw_metrics = evaluate_models(raw_models, X_te_raw, y_te_raw)
    logging.info(f"Raw dataset metrics:\n{raw_metrics}")

    # --- Imputed dataset ---
    imp_df = load_imputed_data()
    X_imp = imp_df[FEATURE_COLS]
    y_imp = np.log1p(imp_df["Cost_clean"])

    X_tr_imp, X_te_imp, y_tr_imp, y_te_imp = train_test_split(
        X_imp, y_imp, test_size=0.2, random_state=42
    )

    imp_models = train_models(X_tr_imp, y_tr_imp)
    imputed_metrics = evaluate_models(imp_models, X_te_imp, y_te_imp)
    logging.info(f"Imputed dataset metrics:\n{imputed_metrics}")

    # --- Plot comparison ---
    logging.info("Plotting before vs after imputation comparison...")
    plot_before_after_comparison(raw_metrics, imputed_metrics)


if __name__ == "__main__":
    main()
