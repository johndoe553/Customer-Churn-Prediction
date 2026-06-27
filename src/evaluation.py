"""
evaluation.py — Evaluate model performance and generate evaluation artefacts.

Produces:
  - Per-model metrics (accuracy, precision, recall, F1, ROC-AUC).
  - model_comparison.csv with all models.
  - Confusion matrix plot (final model).
  - ROC curve plot (final model).
  - Feature importance bar chart (final model).
  - Model comparison bar chart.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit compatibility
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from typing import Dict

from src.config import (
    CONFUSION_MATRIX_PNG,
    ROC_CURVE_PNG,
    FEATURE_IMPORTANCE_PNG,
    MODEL_COMPARISON_CSV,
    CHART_DPI,
    CHART_STYLE,
    COLOUR_PALETTE,
)
from src.utils import get_logger, ensure_directory

logger = get_logger(__name__)


def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict:
    """
    Compute evaluation metrics on the test set.

    Parameters
    ----------
    pipeline : fitted pipeline
    X_test : pd.DataFrame
    y_test : pd.Series
    model_name : str

    Returns
    -------
    dict
        Evaluation metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_proba), 4)),
    }

    logger.info(
        "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )

    return metrics


def evaluate_all_models(
    trained_pipelines: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_results: Dict,
) -> pd.DataFrame:
    """
    Evaluate all trained models and merge with cross-validation results.

    Parameters
    ----------
    trained_pipelines : dict
    X_test : pd.DataFrame
    y_test : pd.Series
    cv_results : dict

    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics.
    """
    rows = []
    for name, pipeline in trained_pipelines.items():
        test_metrics = evaluate_model(pipeline, X_test, y_test, name)
        cv_metrics = cv_results.get(name, {})
        merged = {**test_metrics, **cv_metrics}
        rows.append(merged)

    comparison_df = pd.DataFrame(rows)
    ensure_directory(MODEL_COMPARISON_CSV.parent)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    logger.info("Model comparison saved to %s", MODEL_COMPARISON_CSV)

    return comparison_df


def get_classification_report_text(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> str:
    """Return a formatted classification report string."""
    y_pred = pipeline.predict(X_test)
    return classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"]
    )


def plot_confusion_matrix(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path=None,
) -> None:
    """
    Generate and save a labelled confusion-matrix heatmap.
    """
    save_path = save_path or CONFUSION_MATRIX_PNG
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.set_style(CHART_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    ensure_directory(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curve(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path=None,
) -> None:
    """
    Generate and save a ROC curve with AUC score.
    """
    save_path = save_path or ROC_CURVE_PNG
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    sns.set_style(CHART_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        color=COLOUR_PALETTE["primary"],
        lw=2,
        label=f"{model_name} (AUC = {auc_score:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLOUR_PALETTE["neutral"],
        lw=1,
        label="Random Baseline",
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    ensure_directory(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", save_path)


def plot_feature_importance(
    pipeline,
    feature_names: list,
    model_name: str,
    top_n: int = 15,
    save_path=None,
) -> None:
    """
    Plot feature importance from the fitted classifier.
    Works with tree-based models and logistic regression.
    """
    save_path = save_path or FEATURE_IMPORTANCE_PNG
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning("Classifier does not expose feature importances. Skipping plot.")
        return

    # Align feature names with importances
    n_features = len(importances)
    if len(feature_names) != n_features:
        logger.warning(
            "Feature name count (%d) != importance count (%d). Using indices.",
            len(feature_names),
            n_features,
        )
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    sns.set_style(CHART_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        y="Feature",
        x="Importance",
        color=COLOUR_PALETTE["primary"],
        ax=ax,
    )
    ax.set_title(
        f"Top {top_n} Feature Importances — {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("")

    ensure_directory(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", save_path)


def plot_model_comparison(comparison_df: pd.DataFrame, save_path=None) -> None:
    """
    Create a grouped bar chart comparing models across key metrics.
    """
    from src.config import OUTPUTS_DIR

    save_path = save_path or (OUTPUTS_DIR / "model_comparison_chart.png")

    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    available = [m for m in metrics_to_plot if m in comparison_df.columns]

    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=available,
        var_name="Metric",
        value_name="Score",
    )

    sns.set_style(CHART_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model",
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    ensure_directory(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", save_path)


def generate_model_selection_rationale(
    comparison_df: pd.DataFrame,
    best_model_name: str,
) -> str:
    """
    Generate a plain-English rationale for model selection.

    Parameters
    ----------
    comparison_df : pd.DataFrame
    best_model_name : str

    Returns
    -------
    str
        Human-readable rationale.
    """
    best_row = comparison_df[comparison_df["model"] == best_model_name].iloc[0]

    rationale = (
        f"**{best_model_name}** was selected as the final model based on a composite "
        f"evaluation across multiple metrics.\n\n"
        f"- **F1-Score**: {best_row.get('f1_score', 'N/A'):.4f} — balances precision "
        f"and recall for the minority churn class.\n"
        f"- **Recall**: {best_row.get('recall', 'N/A'):.4f} — the proportion of actual "
        f"churners correctly identified.\n"
        f"- **ROC-AUC**: {best_row.get('roc_auc', 'N/A'):.4f} — the model's ability to "
        f"distinguish between churners and non-churners.\n"
        f"- **Accuracy**: {best_row.get('accuracy', 'N/A'):.4f} — overall correctness "
        f"(note: can be misleading for imbalanced datasets).\n\n"
        f"F1-score was the primary selection criterion because accuracy alone can "
        f"overstate performance when churn is the minority class. A model that predicts "
        f"'No Churn' for everyone could achieve ~73% accuracy but miss all actual "
        f"churners — which defeats the purpose of the tool.\n\n"
        f"When two models scored within 0.02 of each other, the more interpretable "
        f"model was preferred to support transparent decision-making."
    )

    return rationale
