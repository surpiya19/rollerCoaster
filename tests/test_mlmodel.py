import pytest
import pandas as pd
import numpy as np
from scripts.mlmodel import load_data, train_models, evaluate_models


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Cost": [1_000_000, 2_000_000, 1_500_000, 2_500_000],
            "Gforce": [3.5, 4.0, 2.8, 3.9],
            "Speed_mph": [60, 80, 70, 90],
        }
    )


def test_load_data(tmp_path, sample_df):
    path = tmp_path / "clean.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(path)
    assert "Cost" in df.columns
    assert df.shape[0] == 4


def test_train_and_evaluate(sample_df):
    X = sample_df[["Gforce", "Speed_mph"]]
    y = np.log1p(sample_df["Cost"])
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = train_models(X_train, y_train)
    # All 3 models exist
    assert set(models.keys()) == {"Linear Regression", "Random Forest", "XGBoost"}

    results = evaluate_models(models, X_test, y_test)
    assert not results.empty
    assert all(col in results.columns for col in ["Model", "MAE", "RMSE", "R2"])
    # Predictions should have correct length
    for name, model in models.items():
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)


def test_single_sample_warning():
    df = pd.DataFrame({"Cost": [1_000_000], "Gforce": [3.5], "Speed_mph": [60]})
    X = df[["Gforce", "Speed_mph"]]
    y = np.log1p(df["Cost"])
    models = train_models(X, y)
    results = evaluate_models(models, X, y)
    # Should return a results dataframe even with single row
    assert not results.empty
