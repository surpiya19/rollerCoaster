import pytest
import polars as pl
from scripts.coaster_analysis import load_data, clean_data, filter_data


# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def sample_df():
    data = {
        "coaster_name": ["Ride1", "Ride2", "Ride1"],
        "Location": ["ParkA", "ParkB", "ParkA"],
        "Status": ["Operating", "Closed", "Operating"],
        "Manufacturer": ["MakerA", "MakerB", "MakerA"],
        "year_introduced": [2001, 1999, 2001],
        "Cost": ["$1,000,000", "approx 2,000,000 USD", "$1 million"],
        "latitude": [40.0, 41.0, 40.0],
        "longitude": [-75.0, -76.0, -75.0],
        "Type_Main": ["Steel", "Wood", "Steel"],
        "opening_date_clean": ["2001-05-01", "1999-07-10", "2001-05-01"],
        "speed_mph": [60, 80, 60],
        "height_ft": [100, 80, 100],
        "Inversions_clean": [2, 0, 2],
        "Gforce_clean": [3.5, 2.8, 3.5],
    }
    return pl.DataFrame(data)


# ---------------------------
# Tests
# ---------------------------
def test_load_data(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.write_csv(path)
    df = load_data(path)
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 3


def test_clean_data_removes_duplicates(sample_df):
    df_clean = clean_data(sample_df)
    # Duplicate Ride1 should be removed
    assert df_clean.shape[0] == 2
    assert "Cost" in df_clean.columns


def test_filter_data_subsets(sample_df, tmp_path):
    df_clean = clean_data(sample_df)
    subsets = filter_data(df_clean, subsets_dir=tmp_path)
    # Active subset has only "Operating"
    assert subsets["Active"].shape[0] == 1
    # Fast subset (>100 mph) should be empty
    assert subsets["Fast"].shape[0] == 0
