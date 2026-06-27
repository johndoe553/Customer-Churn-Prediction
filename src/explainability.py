"""
explainability.py — SHAP-based model explanations.

Provides:
  - Global SHAP summary plot.
  - Local SHAP explanation for individual predictions.
  - Auto-selects appropriate explainer based on model type.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import SHAP_SUMMARY_PNG, SHAP_BACKGROUND_PATH, CHART_DPI
from src.utils import get_logger, ensure_directory, save_pickle, load_pickle

logger = get_logger(__name__)


def _get_explainer(classifier, background_data: np.ndarray):
    """
    Select and return the appropriate SHAP explainer.

    - TreeExplainer for tree-based models (Random Forest, XGBoost, Decision Tree).
    - LinearExplainer for linear models (Logistic Regression).
    - KernelExplainer as fallback.

    Parameters
    ----------
    classifier : fitted estimator
    background_data : np.ndarray
        Preprocessed background sample for the explainer.

    Returns
    -------
    shap.Explainer
    """
    class_name = type(classifier).__name__

    if class_name in ("RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier"):
        logger.info("Using TreeExplainer for %s", class_name)
        return shap.TreeExplainer(classifier)
    elif class_name == "LogisticRegression":
        logger.info("Using LinearExplainer for %s", class_name)
        return shap.LinearExplainer(classifier, background_data)
    else:
        logger.info("Using KernelExplainer for %s (fallback)", class_name)
        return shap.KernelExplainer(classifier.predict_proba, background_data)


def compute_shap_values(
    pipeline,
    X_train: pd.DataFrame,
    feature_names: list,
    n_background: int = 100,
):
    """
    Compute global SHAP values for the training data.

    Parameters
    ----------
    pipeline : fitted imblearn Pipeline
    X_train : pd.DataFrame
        Raw training features (before preprocessing).
    feature_names : list of str
        Feature names after preprocessing.
    n_background : int
        Number of background samples for the explainer.

    Returns
    -------
    tuple of (shap_values, explainer, X_transformed)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform training data through the preprocessor
    X_transformed = preprocessor.transform(X_train)

    # Sample background data
    if X_transformed.shape[0] > n_background:
        rng = np.random.RandomState(42)
        indices = rng.choice(X_transformed.shape[0], n_background, replace=False)
        background = X_transformed[indices]
    else:
        background = X_transformed

    # Save background for later use
    save_pickle(background, SHAP_BACKGROUND_PATH)

    # Get explainer
    explainer = _get_explainer(classifier, background)

    # Compute SHAP values (limit samples for speed)
    max_explain = min(500, X_transformed.shape[0])
    X_explain = X_transformed[:max_explain]

    logger.info("Computing SHAP values for %d samples ...", max_explain)
    shap_values = explainer.shap_values(X_explain)

    # For binary classification, some explainers return a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use churn (positive) class

    return shap_values, explainer, X_explain


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    save_path=None,
) -> None:
    """
    Generate and save a global SHAP summary (beeswarm) plot.
    """
    save_path = save_path or SHAP_SUMMARY_PNG
    ensure_directory(save_path.parent)

    fig, ax = plt.subplots(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_size=None,
    )

    plt.title("SHAP Feature Impact on Churn Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary plot saved to %s", save_path)


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top N features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values : np.ndarray
    feature_names : list of str
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Mean_SHAP_Value.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    n = min(len(feature_names), len(mean_abs))
    feature_importance = pd.DataFrame(
        {"Feature": feature_names[:n], "Mean_SHAP_Value": mean_abs[:n]}
    ).sort_values("Mean_SHAP_Value", ascending=False)

    return feature_importance.head(top_n).reset_index(drop=True)


def explain_single_prediction(
    pipeline,
    input_df: pd.DataFrame,
    feature_names: list,
    background_data: np.ndarray = None,
) -> dict:
    """
    Generate a local SHAP explanation for a single prediction.

    Parameters
    ----------
    pipeline : fitted pipeline
    input_df : pd.DataFrame
        Single-row dataframe (raw features).
    feature_names : list of str
    background_data : np.ndarray, optional
        Preprocessed background data (loaded from shap_background.pkl).

    Returns
    -------
    dict
        Keys: shap_values, feature_names, feature_values, base_value.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(input_df)

    if background_data is None:
        try:
            background_data = load_pickle(SHAP_BACKGROUND_PATH)
        except FileNotFoundError:
            logger.warning(
                "SHAP background data not found. Local explanations unavailable."
            )
            return None

    try:
        explainer = _get_explainer(classifier, background_data)
        sv = explainer.shap_values(X_transformed)

        if isinstance(sv, list):
            sv = sv[1]  # churn class

        sv = sv.flatten()

        # Get base value
        if hasattr(explainer, "expected_value"):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0

        n = min(len(feature_names), len(sv))
        return {
            "shap_values": sv[:n],
            "feature_names": feature_names[:n],
            "feature_values": X_transformed.flatten()[:n],
            "base_value": float(base_value),
        }
    except Exception as e:
        logger.error("Error computing local SHAP explanation: %s", e)
        return None


def get_shap_interpretation_text(top_features_df: pd.DataFrame) -> list:
    """
    Generate plain-English interpretations for the top SHAP features.

    Parameters
    ----------
    top_features_df : pd.DataFrame
        Output of get_top_shap_features.

    Returns
    -------
    list of str
        Human-readable interpretations.
    """
    interpretations = []
    feature_explanations = {
        "Contract_Month-to-month": (
            "Month-to-month contracts were associated with higher churn risk. "
            "Customers without long-term commitments may find it easier to leave."
        ),
        "tenure": (
            "Tenure (how long a customer has been with the company) is a strong predictor. "
            "Newer customers tend to churn more than long-standing ones."
        ),
        "MonthlyCharges": (
            "Higher monthly charges were linked to increased churn risk. "
            "Customers paying more may feel they are not getting sufficient value."
        ),
        "TotalCharges": (
            "Total charges reflect the customer's cumulative spending. "
            "Lower totals often correlate with newer, higher-risk customers."
        ),
        "InternetService_Fiber optic": (
            "Fibre optic customers showed higher churn rates, possibly due to "
            "higher costs or service quality expectations."
        ),
        "TechSupport_No": (
            "Customers without tech support were more likely to churn. "
            "Lack of support may lead to unresolved issues and dissatisfaction."
        ),
        "OnlineSecurity_No": (
            "Customers without online security services showed higher churn risk. "
            "Additional services may increase perceived value and stickiness."
        ),
        "PaymentMethod_Electronic check": (
            "Customers using electronic cheque payment showed higher churn, "
            "possibly indicating less engagement with automated billing."
        ),
        "PaperlessBilling_Yes": (
            "Paperless billing was associated with higher churn, possibly reflecting "
            "a more digitally engaged and comparison-shopping customer segment."
        ),
    }

    for _, row in top_features_df.iterrows():
        feature = row["Feature"]
        if feature in feature_explanations:
            interpretations.append(feature_explanations[feature])
        else:
            # Generic interpretation
            clean_name = feature.replace("_", " ").replace("  ", " ")
            interpretations.append(
                f"**{clean_name}** was identified as an important factor in the "
                f"model's churn predictions (mean |SHAP| = {row['Mean_SHAP_Value']:.4f})."
            )

    return interpretations
