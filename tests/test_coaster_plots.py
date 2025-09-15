import pytest
import polars as pl
import pandas as pd
import numpy as np

# ML model plotting functions
import scripts.mlmodel as ml
from scripts.mlmodel import train_models, plot_predictions, xgboost_detailed


# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def sample_df_coaster():
    """Sample Polars DataFrame for coaster_analysis plots"""
    data = {
        "Coaster_Name": ["Ride1", "Ride2", "Ride3"],
        "Location": ["ParkA", "ParkB", "ParkC"],
        "Status": ["Operating", "Operating", "Closed"],
        "Year_Introduced": [2001, 2005, 2010],
        "Opening_Date": ["2001-05-01", "2005-06-10", "2010-07-15"],
        "Speed_mph": [60, 80, 100],
        "Height_ft": [100, 120, 80],
        "Inversions": [2, 3, 0],
        "Gforce": [3.5, 4.0, 2.8],
        "Cost": [1_000_000, 2_000_000, 1_500_000],
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_df_ml():
    """Sample Pandas DataFrame for mlmodel plots"""
    return pd.DataFrame(
        {
            "Cost": [1_000_000, 2_000_000, 1_500_000, 2_500_000],
            "Gforce": [3.5, 4.0, 2.8, 3.9],
            "Speed_mph": [60, 80, 70, 90],
        }
    )


# ---------------------------
# Coaster Analysis Plot Tests
# ---------------------------
@pytest.mark.parametrize(
    "plot_func, filename",
    [
        ("plot_speed_distribution", "speed_distribution_test.png"),
        ("plot_yearly_trend", "yearly_trend_test.png"),
        ("plot_height_vs_speed", "height_vs_speed_test.png"),
        ("plot_cost_vs_speed", "cost_vs_speed_test.png"),
        ("plot_correlation_heatmap", "correlation_heatmap_test.png"),
    ],
)
def test_coaster_analysis_plots(tmp_path, sample_df_coaster, plot_func, filename):
    save_path = tmp_path / filename
    func = globals().get(plot_func)
    if not func:
        # Import dynamically if not in globals
        func = getattr(
            __import__("scripts.coaster_analysis", fromlist=[plot_func]), plot_func
        )
    func(sample_df_coaster, save_path=save_path)
    assert save_path.exists()


# ---------------------------
# ML Model Plot Tests
# ---------------------------
def test_mlmodel_plots(tmp_path, sample_df_ml):
    X = sample_df_ml[["Gforce", "Speed_mph"]]
    y = np.log1p(sample_df_ml["Cost"])
    models = train_models(X, y)

    # Save original PLOTS_DIR and monkey-patch for testing
    original_dir = ml.PLOTS_DIR
    try:
        ml.PLOTS_DIR = tmp_path

        # Test plot_predictions
        plot_predictions(models, X, y, y)
        assert (tmp_path / "predictions.png").exists()

        # Test xgboost_detailed
        xgboost_detailed(models, X, y)
        assert (tmp_path / "xgboost_predictions.png").exists()
    finally:
        # Restore original PLOTS_DIR
        ml.PLOTS_DIR = original_dir
