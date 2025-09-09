#!/usr/bin/env python3
# coding: utf-8
"""
coaster_analysis.py
-------------------

Production-ready pipeline for cleaning, analyzing, and visualizing roller coaster data.

Features:
1. Data cleaning (date, cost, numeric columns, duplicates)
2. Summary stats
3. Exploratory plots:
   - Speed distribution
   - Top manufacturers (bar chart)
   - Trend of coaster introductions over years
   - Height vs Speed scatter
   - Cost vs Speed scatter
   - Correlation heatmap
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
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SUBSETS_DIR = Path("subsets")
SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

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
    logging.info("Selecting relevant columns...")
    coaster = df[
        [
            "coaster_name",
            "Location",
            "Status",
            "Manufacturer",
            "year_introduced",
            "Cost",
            "latitude",
            "longitude",
            "Type_Main",
            "opening_date_clean",
            "speed_mph",
            "height_ft",
            "Inversions_clean",
            "Gforce_clean",
        ]
    ].clone()

    logging.info("Parsing Opening_Date...")
    coaster = coaster.with_columns(
        pl.col("opening_date_clean")
        .str.to_datetime(strict=False)
        .dt.date()
        .alias("opening_date_clean")
    )

    logging.info("Casting Speed_mph and Height_ft to float...")
    coaster = coaster.with_columns(
        [
            pl.col("speed_mph").cast(pl.Float64, strict=False),
            pl.col("height_ft").cast(pl.Float64, strict=False),
        ]
    )

    logging.info("Cleaning Cost column...")
    coaster = coaster.with_columns(
        pl.col("Cost")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\[.*?\]|\(.*?\)|approx.*|about.*|rebuild.*|est.*", "")
        .str.replace_all(r"us\$|usd|\$", " USD ")
        .str.replace_all(r"£|gbp", " GBP ")
        .str.replace_all(r"€|eur", " EUR ")
        .str.replace_all(r"¥|yen|jpy", " JPY ")
        .str.replace_all(r"rmb|cny", " CNY ")
        .str.replace_all(r"sek", " SEK ")
        .str.replace_all(r"aud|a\$", " AUD ")
        .str.replace_all(r"cad|c\$", " CAD ")
        .str.replace_all(r" billion", "e9")
        .str.replace_all(r" million", "e6")
        .str.replace_all(r"\bm\b", "e6")
        .str.replace_all(r"(\d)\.(\d{3})(\D|$)", r"\1,\2\3")
        .str.extract(r"([\d\.,]+e\d+|[\d\.,]+)", 1)
        .str.replace_all(",", "")
        .alias("Cost_Clean")
    ).with_columns(pl.col("Cost_Clean").cast(pl.Float64, strict=False))

    logging.info("Renaming columns...")
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

    logging.info("Dropping duplicates...")
    coaster = coaster.unique(
        subset=["Coaster_Name", "Location", "Opening_Date", "Speed_mph"]
    )

    return coaster


# -------------------------
# FILTERING FUNCTIONS
# -------------------------
def filter_data(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    logging.info("Creating filtered subsets...")

    subsets = {
        "Active": df.filter(pl.col("Status") == "Operating"),
        "Modern": df.filter(pl.col("Year_Introduced") >= 2000),
        "Fast": df.filter(pl.col("Speed_mph") >= 70),
        "Tall_inversions": df.filter(
            (pl.col("Height_ft") >= 200) & (pl.col("Inversions") > 0)
        ),
        "Expensive": df.filter(pl.col("Cost") >= 50_000_000),
    }

    for name, subset in subsets.items():
        logging.info(f"{name} subset → {len(subset)} rows")
        subset_path = SUBSETS_DIR / f"{name}_subset.csv"
        subset.write_csv(subset_path)
        logging.info(f"Saved → {subset_path}")

    return subsets


# -------------------------
# ANALYSIS & VISUALS
# -------------------------
def summary_stats(df: pl.DataFrame):
    logging.info("Summary statistics for numeric columns:")
    print(
        df.select(["Speed_mph", "Height_ft", "Cost", "Inversions", "Gforce"]).describe()
    )


def plot_speed_distribution(df: pl.DataFrame):
    logging.info("Plotting speed distribution...")
    speed_data = df.select("Speed_mph").drop_nulls().to_numpy().flatten()

    mean_speed: float = float(speed_data.mean())
    median_speed: float = float(np.median(speed_data))
    std_speed: float = float(speed_data.std())

    plt.figure(figsize=(10, 6))
    plt.hist(
        speed_data,
        bins=25,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        label="Speed Data",
    )

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
    plt.savefig(PLOTS_DIR / "speed_distribution.png")
    plt.close()


def plot_yearly_trend(df: pl.DataFrame):
    logging.info("Plotting yearly trend of coaster introductions...")
    yearly = (
        df.group_by("Year_Introduced")
        .agg(pl.len().alias("Count"))
        .drop_nulls()
        .sort("Year_Introduced")
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=yearly.to_pandas(),
        x="Year_Introduced",
        y="Count",
        marker="o",
        label="Introductions",
    )
    plt.title("Number of Coasters Introduced per Year")
    plt.xlabel("Year Introduced")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(PLOTS_DIR / "yearly_trend.png")
    plt.close()


def plot_height_vs_speed(df: pl.DataFrame):
    logging.info("Plotting Height vs Speed scatter...")
    data = df.select(["Height_ft", "Speed_mph"]).drop_nulls().to_pandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data, x="Height_ft", y="Speed_mph", alpha=0.7, label="Coasters"
    )
    sns.regplot(
        data=data,
        x="Height_ft",
        y="Speed_mph",
        scatter=False,
        color="red",
        label="Trend Line",
    )
    plt.title("Height vs Speed of Roller Coasters")
    plt.xlabel("Height (ft)")
    plt.ylabel("Speed (mph)")
    plt.legend()
    plt.savefig(PLOTS_DIR / "height_vs_speed.png")
    plt.close()


def plot_cost_vs_speed(df: pl.DataFrame):
    logging.info("Plotting Cost vs Speed scatter...")
    data = df.select(["Cost", "Speed_mph"]).drop_nulls().to_pandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="Cost", y="Speed_mph", alpha=0.7, label="Coasters")
    plt.xscale("log")  # cost varies a lot
    plt.title("Cost vs Speed of Roller Coasters")
    plt.xlabel("Cost (log scale)")
    plt.ylabel("Speed (mph)")
    plt.legend()
    plt.savefig(PLOTS_DIR / "cost_vs_speed.png")
    plt.close()


def plot_correlation_heatmap(df: pl.DataFrame):
    logging.info("Plotting correlation heatmap...")
    numeric = (
        df.select(["Speed_mph", "Height_ft", "Cost", "Inversions", "Gforce"])
        .drop_nulls()
        .to_pandas()
    )

    corr = numeric.corr()
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")

    # safer way to grab the colorbar
    cbar: Colorbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Correlation Strength", rotation=270, labelpad=15)

    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()


# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    # Load + Clean
    raw = load_data(RAW_DATA_PATH)
    clean = clean_data(raw)

    logging.info(f"Saving cleaned dataset → {CLEAN_DATA_PATH}")
    clean.write_csv(CLEAN_DATA_PATH)

    # Summary stats (whole dataset)
    summary_stats(clean)

    # Filtering subsets
    subsets = filter_data(clean)

    # Run plots on the full dataset
    plot_speed_distribution(clean)
    plot_yearly_trend(clean)
    plot_height_vs_speed(clean)
    plot_cost_vs_speed(clean)
    plot_correlation_heatmap(clean)

    # Running a plot on one subset for comparison
    # (example: modern coasters only)
    plot_speed_distribution(subsets["Modern"])

    logging.info("Pipeline finished successfully ✅")


if __name__ == "__main__":
    main()
