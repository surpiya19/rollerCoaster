import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

# ==============================
# 1. CLEAN COST COLUMN
# ==============================
data = pd.read_csv("./data/coaster_db_clean.csv")
df = data.copy()
df["Cost_clean"] = df["Cost"]
df.loc[(df["Cost_clean"] < 1e5) | (df["Cost_clean"] > 5e9), "Cost_clean"] = np.nan

# ==============================
# 2. FEATURE ENGINEERING
# ==============================


def add_features(X):
    X = X.copy()
    # Interactions
    X["Height_sq"] = X["Height_ft"] ** 2
    X["Speed_Height"] = X["Speed_mph"] * X["Height_ft"]
    # Binary feature: has inversions
    X["HasInversions"] = (X["Inversions"] > 0).astype(int)
    # Park age feature
    X["ParkAge"] = 2025 - X["Year_Introduced"]
    return X


feature_cols = ["Speed_mph", "Height_ft", "Inversions", "Gforce", "Year_Introduced"]
df_features = add_features(df[feature_cols])

# Keep only rows with known cost for training
train_df = df.loc[df["Cost_clean"].notna()].copy()
train_df_features = add_features(train_df[feature_cols])

X = train_df_features
y = np.log1p(train_df["Cost_clean"])

# ==============================
# 3. TRAIN/VALIDATE WITH GRIDSEARCH
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring="r2", verbose=1)
grid.fit(X_train, y_train)

print("Best R² CV:", grid.best_score_)
print("Best params:", grid.best_params_)

# Evaluate on hold-out test set
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
print("Test R²:", r2_score(y_test, y_pred))

# ==============================
# 4. RETRAIN ON FULL TRAINING DATA
# ==============================
best_rf.fit(X, y)

# ==============================
# 5. PREDICT MISSING COSTS
# ==============================
missing_mask = df["Cost_clean"].isna()
X_missing = add_features(df.loc[missing_mask, feature_cols])

df.loc[missing_mask, "Cost_clean"] = np.expm1(best_rf.predict(X_missing))
df["Cost_imputed"] = missing_mask

print(f"Imputed {missing_mask.sum()} missing cost values.")
