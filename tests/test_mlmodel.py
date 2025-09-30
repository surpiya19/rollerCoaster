import pytest
import pandas as pd
import numpy as np
import scripts.cost_prediction as ml
from scripts.cost_prediction import train_models, evaluate_models, plot_predictions


# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def sample_df_ml():
    return pd.DataFrame(
        {
            "Cost_clean": [1_000_000, 2_000_000, 1_500_000, 2_500_000],
            "Gforce": [3.5, 4.0, 2.8, 3.9],
            "Speed_mph": [60, 80, 70, 90],
            "Height_ft": [100, 120, 80, 110],
            "Inversions": [2, 3, 0, 1],
        }
    )


# ---------------------------
# Tests
# ---------------------------
def test_load_data(tmp_path, sample_df_ml):
    path = tmp_path / "clean.csv"
    sample_df_ml.to_csv(path, index=False)
    df = ml.load_data(path)
    assert "Cost_clean" in df.columns
    assert df.shape[0] == 4


def test_train_and_evaluate(sample_df_ml):
    X = sample_df_ml[["Gforce", "Speed_mph", "Height_ft", "Inversions"]]
    y = np.log1p(sample_df_ml["Cost_clean"])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = train_models(X_train, y_train)
    assert set(models.keys()) == {"Linear Regression", "Random Forest", "XGBoost"}

    results = evaluate_models(models, X_test, y_test)
    assert not results.empty
    assert all(col in results.columns for col in ["Model", "MAE", "RMSE", "R2"])


def test_mlmodel_plots(tmp_path, sample_df_ml):
    X = sample_df_ml[["Gforce", "Speed_mph", "Height_ft", "Inversions"]]
    y = np.log1p(sample_df_ml["Cost_clean"])
    models = train_models(X, y)

    # Monkey-patch PLOTS_DIR
    original_dir = ml.PLOTS_DIR
    ml.PLOTS_DIR = tmp_path
    try:
        plot_predictions(models, X, y, y_full=y)
        # Only check the main predictions file
        assert (tmp_path / "predictions.png").exists()
    finally:
        ml.PLOTS_DIR = original_dir
