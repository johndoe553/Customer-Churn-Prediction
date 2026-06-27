"""
generate_visualisations.py — Generate exploratory data analysis charts.

Usage:
    python scripts/generate_visualisations.py

Produces:
  - churn_distribution.png
  - contract_churn.png
  - tenure_churn.png
  - monthly_charges_churn.png
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import (
    CHURN_DISTRIBUTION_PNG,
    CONTRACT_CHURN_PNG,
    TENURE_CHURN_PNG,
    MONTHLY_CHARGES_CHURN_PNG,
    CHART_DPI,
    CHART_STYLE,
    COLOUR_PALETTE,
)
from src.data_processing import load_dataset, clean_dataset
from src.feature_engineering import engineer_features
from src.utils import get_logger, ensure_directory

logger = get_logger("generate_visualisations")

sns.set_style(CHART_STYLE)


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Churn distribution bar chart with counts and percentages."""
    fig, ax = plt.subplots(figsize=(7, 5))

    counts = df["Churn"].value_counts()
    total = len(df)
    colours = [COLOUR_PALETTE["primary"], COLOUR_PALETTE["danger"]]

    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", width=0.5)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Churn Status", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.2)

    ensure_directory(CHURN_DISTRIBUTION_PNG.parent)
    fig.tight_layout()
    fig.savefig(CHURN_DISTRIBUTION_PNG, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", CHURN_DISTRIBUTION_PNG)


def plot_contract_churn(df: pd.DataFrame) -> None:
    """Churn rate by contract type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100

    ct.plot(
        kind="bar",
        stacked=True,
        color=[COLOUR_PALETTE["primary"], COLOUR_PALETTE["danger"]],
        edgecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Contract Type", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.legend(title="Churn", loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ensure_directory(CONTRACT_CHURN_PNG.parent)
    fig.tight_layout()
    fig.savefig(CONTRACT_CHURN_PNG, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", CONTRACT_CHURN_PNG)


def plot_tenure_churn(df: pd.DataFrame) -> None:
    """Churn rate by tenure group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "TenureGroup" not in df.columns:
        df = engineer_features(df)

    # Order tenure groups logically
    order = ["New", "Early", "Established", "Loyal"]
    ct = pd.crosstab(df["TenureGroup"], df["Churn"], normalize="index") * 100
    ct = ct.reindex(order, axis=0).dropna()

    ct.plot(
        kind="bar",
        stacked=True,
        color=[COLOUR_PALETTE["primary"], COLOUR_PALETTE["danger"]],
        edgecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Tenure Group", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Churn Rate by Tenure Group", fontsize=14, fontweight="bold")
    ax.legend(title="Churn", loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ensure_directory(TENURE_CHURN_PNG.parent)
    fig.tight_layout()
    fig.savefig(TENURE_CHURN_PNG, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", TENURE_CHURN_PNG)


def plot_monthly_charges_churn(df: pd.DataFrame) -> None:
    """Monthly charges distribution by churn status."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, colour in [("No", COLOUR_PALETTE["primary"]), ("Yes", COLOUR_PALETTE["danger"])]:
        subset = df[df["Churn"] == label]["MonthlyCharges"]
        ax.hist(
            subset,
            bins=30,
            alpha=0.6,
            color=colour,
            label=f"Churn = {label}",
            edgecolor="white",
        )

    ax.set_xlabel("Monthly Charges ($)", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    ensure_directory(MONTHLY_CHARGES_CHURN_PNG.parent)
    fig.tight_layout()
    fig.savefig(MONTHLY_CHARGES_CHURN_PNG, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", MONTHLY_CHARGES_CHURN_PNG)


def main():
    """Generate all EDA visualisations."""
    logger.info("=" * 70)
    logger.info("GENERATING EXPLORATORY DATA ANALYSIS VISUALISATIONS")
    logger.info("=" * 70)

    df = load_dataset()
    df = clean_dataset(df)

    plot_churn_distribution(df)
    plot_contract_churn(df)
    plot_tenure_churn(df)
    plot_monthly_charges_churn(df)

    logger.info("\nAll visualisations saved to outputs/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
