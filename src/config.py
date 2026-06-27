"""
config.py — Centralised project configuration.

All paths are relative to the project root so the application is portable
across machines and deployable to Streamlit Community Cloud.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: src/config.py → project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# File paths — data
# ---------------------------------------------------------------------------
DATASET_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# File paths — model artefacts
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.pkl"

# ---------------------------------------------------------------------------
# File paths — outputs
# ---------------------------------------------------------------------------
MODEL_COMPARISON_CSV = OUTPUTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / "confusion_matrix.png"
ROC_CURVE_PNG = OUTPUTS_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PNG = OUTPUTS_DIR / "feature_importance.png"
SHAP_SUMMARY_PNG = OUTPUTS_DIR / "shap_summary.png"
CHURN_DISTRIBUTION_PNG = OUTPUTS_DIR / "churn_distribution.png"
CONTRACT_CHURN_PNG = OUTPUTS_DIR / "contract_churn.png"
TENURE_CHURN_PNG = OUTPUTS_DIR / "tenure_churn.png"
MONTHLY_CHARGES_CHURN_PNG = OUTPUTS_DIR / "monthly_charges_churn.png"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# ---------------------------------------------------------------------------
# Churn risk thresholds (probability %)
# ---------------------------------------------------------------------------
LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Feature lists — columns present in the raw dataset (excluding ID & target)
# ---------------------------------------------------------------------------
RAW_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",        # will be converted to "Yes"/"No"
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# ---------------------------------------------------------------------------
# Engineered feature names
# ---------------------------------------------------------------------------
ENGINEERED_CATEGORICAL = [
    "TenureGroup",
    "HasSecurityServices",
    "HasStreamingServices",
    "IsMonthToMonthContract",
    "HighMonthlyCharge",
    "FamilySupport",
]

ENGINEERED_NUMERIC = [
    "AverageMonthlySpend",
]

# ---------------------------------------------------------------------------
# Final feature sets used for modelling (after engineering, before encoding)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL

# ---------------------------------------------------------------------------
# Chart styling
# ---------------------------------------------------------------------------
CHART_DPI = 150
CHART_STYLE = "whitegrid"
COLOUR_PALETTE = {
    "primary": "#0068C9",
    "secondary": "#83C9FF",
    "danger": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F39C12",
    "neutral": "#95A5A6",
    "background": "#F0F4F8",
}
