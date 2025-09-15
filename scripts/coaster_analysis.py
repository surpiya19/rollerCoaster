#!/usr/bin/env python3
# coding: utf-8

"""
coaster_analysis.py
-------------------

Production-ready pipeline for cleaning, analyzing,
and visualizing roller coaster data.

All existing functions preserved, with fixes for test compatibility.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
from matplotlib.colorbar import Colorbar

# -------------------------
# CONFIG
# -------------------------
RAW_DATA_PATH = Path("data/coaster_db.csv")
CLEAN_DATA_PATH = Path("data/coaster_db_clean.csv")
PLOTS_DIR = Path("plots")
SUBSETS_DIR = Path("subsets")

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path: Path) -> pl.DataFrame:
    logging.info(f"Loading dataset from {path}...")
    return pl.read_csv(path)


# -------------------------
# CLEANING FUNCTIONS
# -------------------------
def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    logging.info("Cleaning coaster dataset...")

    # ------------------------
    # Ensure all required columns exist
    # ------------------------
    required_columns = {
        "coaster_name": "",
        "Location": "",
        "Status": "Unknown",
        "Manufacturer": "Unknown",
        "year_introduced": 0,
        "Cost": 0.0,
        "latitude": 0.0,
        "longitude": 0.0,
        "Type_Main": "",
        "opening_date_clean": None,
        "speed_mph": None,
        "height_ft": None,
        "Inversions_clean": 0,
        "Gforce_clean": 0.0,
    }

    for col, default in required_columns.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    # ------------------------
    # Select relevant columns
    # ------------------------
    coaster = df[list(required_columns.keys())].clone()

    # ------------------------
    # Parse dates
    # ------------------------
    coaster = coaster.with_columns(
        pl.col("opening_date_clean")
        .str.to_datetime(strict=False)
        .dt.date()
        .alias("opening_date_clean")
    )

    # ------------------------
    # Cast numeric columns
    # ------------------------
    coaster = coaster.with_columns(
        [
            pl.col("speed_mph").cast(pl.Float64, strict=False),
            pl.col("height_ft").cast(pl.Float64, strict=False),
            pl.col("Inversions_clean").cast(pl.Int64, strict=False),
            pl.col("Gforce_clean").cast(pl.Float64, strict=False),
        ]
    )

    # ------------------------
    # Clean Cost column
    # ------------------------
    coaster = coaster.with_columns(
        pl.col("Cost")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\[.*?\]|\(.*?\)|approx.*|about.*|rebuild.*|est.*", "")
        .str.replace_all(r"us\$|usd|\$", "")
        .str.replace_all(r"£|gbp", "")
        .str.replace_all(r"€|eur", "")
        .str.replace_all(r"¥|yen|jpy", "")
        .str.replace_all(r"rmb|cny", "")
        .str.replace_all(r"sek", "")
        .str.replace_all(r"aud|a\$", "")
        .str.replace_all(r"cad|c\$", "")
        .str.replace_all(r" billion", "e9")
        .str.replace_all(r" million", "e6")
        .str.replace_all(r"\bm\b", "e6")
        .str.replace_all(r"(\d)\.(\d{3})(\D|$)", r"\1,\2\3")
        .str.replace_all(",", "")
        .str.extract(r"([\d\.,]+e\d+|[\d\.,]+)", 1)
        .alias("Cost_Clean")
    ).with_columns(pl.col("Cost_Clean").cast(pl.Float64, strict=False))

    # ------------------------
    # Rename columns
    # ------------------------
    coaster = coaster.drop("Cost").rename(
        {
            "coaster_name": "Coaster_Name",
            "year_introduced": "Year_Introduced",
            "opening_date_clean": "Opening_Date",
            "speed_mph": "Speed_mph",
            "height_ft": "Height_ft",
            "Inversions_clean": "Inversions",
            "Gforce_clean": "Gforce",
            "Cost_Clean": "Cost",
        }
    )

    # ------------------------
    # Drop duplicates
    # ------------------------
    coaster = coaster.unique(
        subset=["Coaster_Name", "Location", "Opening_Date", "Speed_mph"]
    )

    return coaster


# -------------------------
# FILTERING FUNCTIONS
# -------------------------
def filter_data(
    df: pl.DataFrame, subsets_dir: Path = SUBSETS_DIR
) -> dict[str, pl.DataFrame]:
    """
    Creates subsets and writes CSVs to disk.
    Returns a dictionary of {subset_name: Polars DataFrame}.
    """
    subsets = {}

    subsets_dir.mkdir(parents=True, exist_ok=True)

    # Active coasters
    if "Status" in df.columns:
        subsets["Active"] = df.filter(
            pl.col("Status").str.to_lowercase() == "operating"
        )
    else:
        subsets["Active"] = pl.DataFrame(schema=df.schema)

    # Modern coasters
    if "Opening_Date" in df.columns:
        subsets["Modern"] = df.filter(pl.col("Opening_Date").dt.year() > 2000)
    else:
        subsets["Modern"] = pl.DataFrame(schema=df.schema)

    # Fast coasters
    if "Speed_mph" in df.columns:
        subsets["Fast"] = df.filter(pl.col("Speed_mph") > 100)
    else:
        subsets["Fast"] = pl.DataFrame(schema=df.schema)

    # Tall with inversions
    if {"Height_ft", "Inversions"}.issubset(df.columns):
        subsets["Tall_inversions"] = df.filter(
            (pl.col("Height_ft") > 30) & (pl.col("Inversions") > 0)
        )
    else:
        subsets["Tall_inversions"] = pl.DataFrame(schema=df.schema)

    # Expensive coasters
    if "Cost" in df.columns:
        subsets["Expensive"] = df.filter(pl.col("Cost") > 1_000_000)
    else:
        subsets["Expensive"] = pl.DataFrame(schema=df.schema)

    # Write CSVs
    for name, subset_df in subsets.items():
        subset_df.write_csv(subsets_dir / f"{name}.csv")

    return subsets


# -------------------------
# ANALYSIS & VISUALS
# -------------------------
def summary_stats(df: pl.DataFrame):
    logging.info("Summary statistics for numeric columns:")
    print(
        df.select(["Speed_mph", "Height_ft", "Cost", "Inversions", "Gforce"]).describe()
    )


def plot_speed_distribution(df: pl.DataFrame, save_path: Path | None = None):
    logging.info("Plotting speed distribution...")
    if save_path is None:
        save_path = PLOTS_DIR / "speed_distribution.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    speed_data = df.select("Speed_mph").drop_nulls().to_numpy().flatten()
    mean_speed = float(speed_data.mean()) if speed_data.size else 0
    median_speed = float(np.median(speed_data)) if speed_data.size else 0

    plt.figure(figsize=(10, 6))
    plt.hist(speed_data, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    plt.axvline(
        mean_speed,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean = {mean_speed:.2f}",
    )
    plt.axvline(
        median_speed,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"Median = {median_speed:.2f}",
    )
    plt.title("Distribution of Roller Coaster Speeds")
    plt.xlabel("Speed (mph)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_yearly_trend(df: pl.DataFrame, save_path: Path | None = None):
    logging.info("Plotting yearly trend of coaster introductions...")
    if save_path is None:
        save_path = PLOTS_DIR / "yearly_trend.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    yearly = (
        df.group_by("Year_Introduced")
        .agg(pl.count().alias("Count"))
        .sort("Year_Introduced")
        .drop_nulls()
    )
    sns.lineplot(data=yearly.to_pandas(), x="Year_Introduced", y="Count", marker="o")
    plt.title("Number of Coasters Introduced per Year")
    plt.xlabel("Year Introduced")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()


def plot_height_vs_speed(df: pl.DataFrame, save_path: Path | None = None):
    if save_path is None:
        save_path = PLOTS_DIR / "height_vs_speed.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Plotting Height vs Speed scatter...")
    data = df.select(["Height_ft", "Speed_mph"]).drop_nulls().to_pandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Height_ft", y="Speed_mph", alpha=0.7)
    sns.regplot(data=data, x="Height_ft", y="Speed_mph", scatter=False, color="red")
    plt.title("Height vs Speed of Roller Coasters")
    plt.xlabel("Height (ft)")
    plt.ylabel("Speed (mph)")
    plt.savefig(save_path)
    plt.close()


def plot_cost_vs_speed(df: pl.DataFrame, save_path: Path | None = None):
    if save_path is None:
        save_path = PLOTS_DIR / "cost_vs_speed.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Plotting Cost vs Speed scatter...")
    data = df.select(["Cost", "Speed_mph"]).drop_nulls().to_pandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Cost", y="Speed_mph", alpha=0.7)
    plt.xscale("log")
    plt.title("Cost vs Speed of Roller Coasters")
    plt.xlabel("Cost (log scale)")
    plt.ylabel("Speed (mph)")
    plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(df: pl.DataFrame, save_path: Path | None = None):
    if save_path is None:
        save_path = PLOTS_DIR / "correlation_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Plotting correlation heatmap...")
    numeric = (
        df.select(["Speed_mph", "Height_ft", "Cost", "Inversions", "Gforce"])
        .drop_nulls()
        .to_pandas()
    )
    corr = numeric.corr()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    cbar: Colorbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Correlation Strength", rotation=270, labelpad=15)
    plt.savefig(save_path)
    plt.close()


# -------------------------
# MAIN PIPELINE
# -------------------------
def main(
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_DATA_PATH,
    plots_dir: Path = PLOTS_DIR,
    subsets_dir: Path = SUBSETS_DIR,
):

    # Load + Clean
    raw = load_data(raw_path)
    clean = clean_data(raw)
    logging.info(f"Saving cleaned dataset → {clean_path}")
    clean.write_csv(clean_path)

    # Summary stats
    summary_stats(clean)

    # Filtering subsets
    subsets = filter_data(clean, subsets_dir=subsets_dir)

    # Run plots on the full dataset
    plot_speed_distribution(clean, save_path=plots_dir / "speed_distribution.png")
    plot_yearly_trend(clean, save_path=plots_dir / "yearly_trend.png")
    plot_height_vs_speed(clean, save_path=plots_dir / "height_vs_speed.png")
    plot_cost_vs_speed(clean, save_path=plots_dir / "cost_vs_speed.png")
    plot_correlation_heatmap(clean, save_path=plots_dir / "correlation_heatmap.png")

    # Example: modern coasters only
    if "Modern" in subsets:
        plot_speed_distribution(
            subsets["Modern"], save_path=plots_dir / "modern_speed_distribution.png"
        )

    logging.info("Pipeline finished successfully ✅")


if __name__ == "__main__":
    main()
