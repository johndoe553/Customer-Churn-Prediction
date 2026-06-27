"""
data_processing.py — Load, clean, and validate the Telco Customer Churn dataset.

Key responsibilities:
  1. Load CSV and perform basic validation.
  2. Remove duplicate rows.
  3. Strip whitespace from string columns.
  4. Convert SeniorCitizen (0/1) → categorical ("No"/"Yes").
  5. Safely convert TotalCharges to float (handle blanks).
  6. Impute missing numeric values with median.
  7. Impute missing categorical values with mode.
  8. Encode target column to binary (1 = Churn).
"""

import pandas as pd
import numpy as np

from src.config import DATASET_PATH, TARGET_COLUMN, ID_COLUMN
from src.utils import get_logger

logger = get_logger(__name__)


def load_dataset(path=None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV file.

    Parameters
    ----------
    path : str or Path, optional
        Override default dataset path.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    data_path = path or DATASET_PATH
    logger.info("Loading dataset from %s", data_path)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it in the data/ directory."
        )

    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw dataset.

    Steps:
      - Remove duplicates.
      - Strip whitespace from object columns.
      - Convert SeniorCitizen to categorical string.
      - Fix TotalCharges blanks → NaN → float.
      - Impute missing numeric values with median.
      - Impute missing categorical values with mode.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (target still as string).
    """
    df = df.copy()
    initial_rows = len(df)

    # ----- Remove duplicates -----
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info("Removed %d duplicate rows", removed)

    # ----- Strip whitespace in object columns -----
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ----- Convert SeniorCitizen 0/1 → "No"/"Yes" -----
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        logger.info("Converted SeniorCitizen to categorical (No/Yes)")

    # ----- Fix TotalCharges -----
    if "TotalCharges" in df.columns:
        # Replace empty / whitespace-only strings with NaN
        df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_missing = df["TotalCharges"].isna().sum()
        if n_missing > 0:
            median_val = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
            logger.info(
                "Imputed %d missing TotalCharges values with median (%.2f)",
                n_missing,
                median_val,
            )

    # ----- Impute remaining missing numeric values -----
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed missing %s with median (%.2f)", col, median_val)

    # ----- Impute remaining missing categorical values -----
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Imputed missing %s with mode (%s)", col, mode_val)

    logger.info("Cleaning complete. Final shape: %s", df.shape)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target column from 'Yes'/'No' to 1/0.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with string target.

    Returns
    -------
    pd.DataFrame
        Dataframe with binary target.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains unexpected values. "
            "Expected only 'Yes' and 'No'."
        )

    logger.info(
        "Target encoded — Churn=1: %d (%.1f%%), Churn=0: %d (%.1f%%)",
        df[TARGET_COLUMN].sum(),
        100 * df[TARGET_COLUMN].mean(),
        (df[TARGET_COLUMN] == 0).sum(),
        100 * (1 - df[TARGET_COLUMN].mean()),
    )
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Separate features and target, dropping the customer ID.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, target-encoded dataframe.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        X (features) and y (target).
    """
    cols_to_drop = [col for col in [ID_COLUMN, TARGET_COLUMN] if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN]
    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary dictionary for dashboard display.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (target may be string or int).

    Returns
    -------
    dict
        Keys: n_customers, n_features, churn_rate, churn_count, non_churn_count.
    """
    target = df[TARGET_COLUMN]
    # Handle both string and numeric target
    if target.dtype == "object":
        churn_count = (target == "Yes").sum()
    else:
        churn_count = int(target.sum())

    n_customers = len(df)
    non_churn_count = n_customers - churn_count
    churn_rate = churn_count / n_customers if n_customers > 0 else 0.0

    return {
        "n_customers": n_customers,
        "n_features": df.shape[1] - 2,  # exclude ID and target
        "churn_rate": round(churn_rate, 4),
        "churn_count": churn_count,
        "non_churn_count": non_churn_count,
    }
