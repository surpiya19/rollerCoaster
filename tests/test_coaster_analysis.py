import pytest
import polars as pl

from scripts import coaster_analysis as ca


@pytest.fixture
def sample_df(tmp_path):
    """Create a small sample dataset for testing cleaning/filtering."""
    data = pl.DataFrame(
        {
            "coaster_name": ["Test Coaster", "Old Coaster", "Fast One"],
            "Location": ["Park A", "Park B", "Park C"],
            "Status": ["Operating", "Closed", "Operating"],
            "Manufacturer": ["Maker1", "Maker2", "Maker3"],
            "year_introduced": [2010, 1995, 2021],
            "Cost": ["$10 million", "$5 million", "$100 million"],
            "latitude": [10.0, 20.0, 30.0],
            "longitude": [100.0, 200.0, 300.0],
            "Type_Main": ["Steel", "Wood", "Steel"],
            "opening_date_clean": ["2010-01-01", "1995-05-05", "2021-08-08"],
            "speed_mph": [50, 40, 80],
            "height_ft": [100, 50, 210],
            "Inversions_clean": [1, 0, 2],
            "Gforce_clean": [3.5, 3.0, 4.0],
        }
    )
    return data


def test_clean_data_removes_duplicates_and_casts(sample_df):
    clean = ca.clean_data(sample_df)

    # 1) Columns should have been renamed
    assert "Coaster_Name" in clean.columns
    assert "Speed_mph" in clean.columns

    # 2) Cost column should be numeric
    assert clean["Cost"].dtype == pl.Float64

    # 3) No duplicate rows should remain
    unique_rows = clean.unique(
        subset=["Coaster_Name", "Location", "Opening_Date", "Speed_mph"]
    )
    assert len(unique_rows) == len(clean)


def test_filter_data_creates_expected_subsets(sample_df):
    clean = ca.clean_data(sample_df)
    subsets = ca.filter_data(clean)

    # 1) Should return a dictionary with correct keys
    expected_keys = {"Active", "Modern", "Fast", "Tall_inversions", "Expensive"}
    assert set(subsets.keys()) == expected_keys

    # 2) 'Fast' subset should only include rows >= 70 mph
    fast_subset = subsets["Fast"]
    assert all(fast_subset["Speed_mph"] >= 70)

    # 3) 'Modern' subset should only include coasters >= 2000
    modern_subset = subsets["Modern"]
    assert all(modern_subset["Year_Introduced"] >= 2000)


def test_plot_functions_create_files(sample_df, tmp_path, monkeypatch):
    clean = ca.clean_data(sample_df)

    # Override PLOTS_DIR so we don't write into the real folder
    monkeypatch.setattr(ca, "PLOTS_DIR", tmp_path)

    ca.plot_speed_distribution(clean)
    ca.plot_yearly_trend(clean)
    ca.plot_height_vs_speed(clean)
    ca.plot_cost_vs_speed(clean)
    ca.plot_correlation_heatmap(clean)

    # Check that files were actually created
    plot_files = list(tmp_path.glob("*.png"))
    assert len(plot_files) >= 5


def test_load_data_reads_csv(tmp_path):
    # Create a fake CSV
    csv_path = tmp_path / "test.csv"
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df.write_csv(csv_path)

    loaded = ca.load_data(csv_path)

    assert isinstance(loaded, pl.DataFrame)
    assert loaded.shape == (2, 2)
    assert set(loaded.columns) == {"a", "b"}
