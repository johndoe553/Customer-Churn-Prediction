"""
feature_engineering.py — Create interpretable engineered features.

This module provides a single `engineer_features` function used by both
the training pipeline and the Streamlit prediction interface, ensuring
consistency between training and inference.

Engineered features:
  1. TenureGroup         — categorical bucketing of tenure.
  2. AverageMonthlySpend — TotalCharges / max(tenure, 1).
  3. HasSecurityServices  — Yes if any security-related service is active.
  4. HasStreamingServices — Yes if any streaming service is active.
  5. IsMonthToMonthContract — Yes/No flag.
  6. HighMonthlyCharge    — Yes if MonthlyCharges >= dataset median.
  7. FamilySupport        — Yes if customer has Partner or Dependents.
"""

import pandas as pd
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

# Default median monthly charge (updated at training time if needed).
# This is a fallback; the training script should pass the actual median.
_DEFAULT_MONTHLY_CHARGE_MEDIAN = 70.35


def engineer_features(
    df: pd.DataFrame,
    monthly_charge_median: float = None,
) -> pd.DataFrame:
    """
    Add all engineered features to the dataframe.

    This function is idempotent: if an engineered column already exists,
    it will be overwritten with a fresh calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw features (cleaned).
    monthly_charge_median : float, optional
        Median MonthlyCharges from the training set.  Used for the
        HighMonthlyCharge flag.  Falls back to a sensible default
        if not provided (e.g., during single-record prediction).

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered columns.
    """
    df = df.copy()

    if monthly_charge_median is None:
        monthly_charge_median = _DEFAULT_MONTHLY_CHARGE_MEDIAN

    # ----- 1. TenureGroup -----
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["New", "Early", "Established", "Loyal"],
    ).astype(str)

    # ----- 2. AverageMonthlySpend -----
    df["AverageMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # ----- 3. HasSecurityServices -----
    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    existing_sec = [c for c in security_cols if c in df.columns]
    if existing_sec:
        df["HasSecurityServices"] = (
            df[existing_sec].isin(["Yes"]).any(axis=1).map({True: "Yes", False: "No"})
        )
    else:
        df["HasSecurityServices"] = "No"

    # ----- 4. HasStreamingServices -----
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    existing_str = [c for c in streaming_cols if c in df.columns]
    if existing_str:
        df["HasStreamingServices"] = (
            df[existing_str].isin(["Yes"]).any(axis=1).map({True: "Yes", False: "No"})
        )
    else:
        df["HasStreamingServices"] = "No"

    # ----- 5. IsMonthToMonthContract -----
    if "Contract" in df.columns:
        df["IsMonthToMonthContract"] = (
            df["Contract"]
            .apply(lambda x: "Yes" if x == "Month-to-month" else "No")
        )
    else:
        df["IsMonthToMonthContract"] = "No"

    # ----- 6. HighMonthlyCharge -----
    df["HighMonthlyCharge"] = (
        df["MonthlyCharges"]
        .apply(lambda x: "Yes" if x >= monthly_charge_median else "No")
    )

    # ----- 7. FamilySupport -----
    partner = df.get("Partner", pd.Series(["No"] * len(df)))
    dependents = df.get("Dependents", pd.Series(["No"] * len(df)))
    df["FamilySupport"] = np.where(
        (partner == "Yes") | (dependents == "Yes"), "Yes", "No"
    )

    logger.info("Engineered %d features", 7)
    return df
