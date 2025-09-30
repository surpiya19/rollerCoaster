import pytest
import polars as pl


# Dynamic imports for plotting functions
def import_plot_func(name):
    return getattr(__import__("scripts.coaster_analysis", fromlist=[name]), name)


@pytest.fixture
def sample_df_coaster():
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
    func = import_plot_func(plot_func)
    func(sample_df_coaster, save_path=save_path)
    assert save_path.exists()
