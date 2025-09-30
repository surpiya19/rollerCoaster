#!/usr/bin/env python3
# coding: utf-8

"""
Impute missing roller coaster construction costs using Random Forest
and save the imputed dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# --- Paths ---
DATA_PATH = Path("./data/coaster_db_clean.csv")
OUTPUT_PATH = Path("./data/coaster_db_imputed.csv")
MODEL_PATH = Path("./models/rf_imputer.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True)


# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Height_sq"] = df["Height_ft"] ** 2
    df["Speed_Height"] = df["Speed_mph"] * df["Height_ft"]
    df["HasInversions"] = (df["Inversions"] > 0).astype(int)
    df["ParkAge"] = 2025 - df["Year_Introduced"]
    numeric_cols = [
        "Gforce",
        "Speed_mph",
        "Height_ft",
        "Inversions",
        "Height_sq",
        "Speed_Height",
        "HasInversions",
        "ParkAge",
    ]
    return df[numeric_cols]


# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(DATA_PATH)

    # Clean Cost
    df["Cost_clean"] = df["Cost"]
    df.loc[(df["Cost_clean"] < 1e5) | (df["Cost_clean"] > 5e9), "Cost_clean"] = np.nan

    # Prepare training data
    train_df = df[df["Cost_clean"].notna()]
    X_train = engineer_features(train_df)
    y_train = np.log1p(train_df["Cost_clean"])

    # Split for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # GridSearch Random Forest
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_tr)

    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_val)
    print("Best RF params:", grid.best_params_)
    print("Holdout R²:", r2_score(y_val, y_pred))

    # Retrain on full training set
    best_rf.fit(X_train, y_train)
    joblib.dump(best_rf, MODEL_PATH)

    # Impute missing costs
    missing_mask = df["Cost_clean"].isna()
    if missing_mask.sum() > 0:
        X_missing = engineer_features(df.loc[missing_mask])
        df.loc[missing_mask, "Cost_clean"] = np.expm1(best_rf.predict(X_missing))
    df["Cost_imputed"] = missing_mask

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved imputed dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
