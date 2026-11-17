"""
01 - Explore the bankruptcy data.

Loads the 43,405 company records and looks at four things before any model:
how many records are bankruptcies in each forecast-horizon group, which ratios
have missing values, how skewed the ratios are, and which ratios differ most
between bankrupt and operating companies. Ends by printing what each of the
64 ratios means.

Run: python 01_explore_data.py
Figures: figures/01_explore_data/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from columns import DEFINITIONS
from common import COLORS, GROUP, TARGET, financial_columns, load_data, readable, save_figure

FIGURES = Path(__file__).stem
N_DISTRIBUTIONS = 12  # ratios shown in the distribution grid


def bankruptcy_rate_by_year(data):
    """Operating and bankrupt records in each forecast-horizon group."""
    counts = data.groupby([GROUP, TARGET]).size().unstack(fill_value=0)
    table = pd.DataFrame({"operating": counts[0], "bankrupt": counts[1]})
    table["total"] = table["operating"] + table["bankrupt"]
    table["bankrupt_pct"] = (table["bankrupt"] / table["total"] * 100).round(2)
    return table


def missing_values(data):
    """Share of missing values per column (only columns with gaps) and rows with any gap."""
    missing = data.isna().sum()
    missing_pct = (missing[missing > 0].sort_values(ascending=False) / len(data) * 100).round(2)
    rows_with_missing = int(data.isna().any(axis=1).sum())
    return missing_pct, rows_with_missing


def median_gap(data):
    """Median of each ratio for operating and bankrupt companies, largest gap first.

    Medians rather than means: the tails are far too heavy for means to say much.
    """
    medians = data.groupby(TARGET)[financial_columns(data)].median().T
    medians.columns = ["operating", "bankrupt"]
    medians["abs_diff"] = (medians["bankrupt"] - medians["operating"]).abs()
    return medians.sort_values("abs_diff", ascending=False)


def plot_rate_by_year(by_year):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(by_year.index.astype(str), by_year["bankrupt_pct"], color=COLORS["bankrupt"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(title="Bankruptcy rate by forecast-horizon group",
           xlabel="Horizon group (1 = five years ahead, 5 = one year ahead)",
           ylabel="Bankrupt records (%)")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, FIGURES, "01_bankruptcy_rate_by_year.png")


def plot_distributions(data):
    """Histograms of the first ratios, clipped to their 1st-99th percentiles."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for column, ax in zip(financial_columns(data)[:N_DISTRIBUTIONS], axes.ravel()):
        values = data[column].dropna()
        low, high = values.quantile([0.01, 0.99])
        ax.hist(values.clip(low, high), bins=50, color=COLORS["operating"])
        ax.set_title(readable(column, width=30), fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Most ratios have extreme tails (plotted between the 1st and 99th percentiles)")
    save_figure(fig, FIGURES, "02_ratio_distributions.png")


def main():
    data = load_data()
    print(f"Records: {len(data):,} | columns: {data.shape[1]} (64 ratios + year + class)")

    by_year = bankruptcy_rate_by_year(data)
    print("\nBankruptcies by forecast-horizon group:")
    print(by_year.to_string())
    plot_rate_by_year(by_year)

    missing_pct, rows_with_missing = missing_values(data)
    print(f"\nColumns with missing values: {len(missing_pct)} of {data.shape[1]}")
    print(f"Rows with at least one missing value: {rows_with_missing:,} "
          f"({rows_with_missing / len(data):.1%})")
    print(missing_pct.head(15).rename("missing %").to_string())

    plot_distributions(data)

    gap = median_gap(data)
    print("\nRatios whose medians differ most between bankrupt and operating companies:")
    print(gap.head(20).round(4).to_string())

    print("\nWhat the 64 ratios mean:")
    for name, formula in DEFINITIONS.items():
        print(f"  {name}: {formula}")
    print(f"\nFigures saved in figures/{FIGURES}/")

    return {
        "shape": data.shape,
        "by_year": by_year,
        "columns_with_missing": len(missing_pct),
        "rows_with_missing": rows_with_missing,
        "missing_pct": missing_pct,
        "median_gap": gap,
    }


if __name__ == "__main__":
    main()
