"""
train_model.py — End-to-end model training pipeline.

Usage:
    python scripts/train_model.py

This script:
  1. Loads and cleans the dataset.
  2. Engineers features.
  3. Splits data (stratified, no leakage).
  4. Trains and cross-validates four classifiers.
  5. Evaluates on the held-out test set.
  6. Selects the best model.
  7. Generates all evaluation artefacts.
  8. Computes and saves SHAP explanations.
  9. Saves all model artefacts to models/.
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    FEATURE_NAMES_PATH,
    MODEL_COMPARISON_CSV,
)
from src.data_processing import (
    load_dataset,
    clean_dataset,
    encode_target,
    prepare_features,
    get_dataset_summary,
)
from src.feature_engineering import engineer_features
from src.model_training import (
    train_all_models,
    select_best_model,
    get_feature_names_from_pipeline,
)
from src.evaluation import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_model_comparison,
    get_classification_report_text,
    generate_model_selection_rationale,
)
from src.explainability import (
    compute_shap_values,
    plot_shap_summary,
    get_top_shap_features,
)
from src.utils import get_logger, save_pickle, save_json

logger = get_logger("train_model")


def main():
    """Execute the full training pipeline."""
    logger.info("=" * 70)
    logger.info("TELECOM CUSTOMER CHURN PREDICTION — MODEL TRAINING")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    logger.info("\n--- Step 1: Loading and cleaning dataset ---")
    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw)
    summary = get_dataset_summary(df_clean)
    logger.info("Dataset summary: %s", summary)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("\n--- Step 2: Feature engineering ---")
    monthly_charge_median = df_clean["MonthlyCharges"].median()
    logger.info("MonthlyCharges median (for HighMonthlyCharge): %.2f", monthly_charge_median)
    df_engineered = engineer_features(df_clean, monthly_charge_median=monthly_charge_median)

    # ------------------------------------------------------------------
    # 3. Encode target and prepare features
    # ------------------------------------------------------------------
    logger.info("\n--- Step 3: Preparing features and target ---")
    df_encoded = encode_target(df_engineered)
    X, y = prepare_features(df_encoded)

    # Identify actual feature columns present
    numeric_features = [c for c in NUMERIC_FEATURES if c in X.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    logger.info("Numeric features (%d): %s", len(numeric_features), numeric_features)
    logger.info("Categorical features (%d): %s", len(categorical_features), categorical_features)

    # ------------------------------------------------------------------
    # 4. Stratified train-test split (BEFORE fitting preprocessor)
    # ------------------------------------------------------------------
    logger.info("\n--- Step 4: Stratified train-test split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train: %d samples | Test: %d samples",
        X_train.shape[0],
        X_test.shape[0],
    )
    logger.info(
        "Train churn rate: %.2f%% | Test churn rate: %.2f%%",
        100 * y_train.mean(),
        100 * y_test.mean(),
    )

    # ------------------------------------------------------------------
    # 5. Train and cross-validate all models
    # ------------------------------------------------------------------
    logger.info("\n--- Step 5: Training and cross-validating models ---")
    trained_pipelines, cv_results, preprocessor = train_all_models(
        X_train,
        y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on test set
    # ------------------------------------------------------------------
    logger.info("\n--- Step 6: Evaluating models on test set ---")
    comparison_df = evaluate_all_models(
        trained_pipelines, X_test, y_test, cv_results
    )
    logger.info("\nModel comparison:\n%s", comparison_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Select best model
    # ------------------------------------------------------------------
    logger.info("\n--- Step 7: Selecting best model ---")
    best_model_name = select_best_model(cv_results)
    best_pipeline = trained_pipelines[best_model_name]

    # Get classification report
    report = get_classification_report_text(best_pipeline, X_test, y_test)
    logger.info("\nClassification report (%s):\n%s", best_model_name, report)

    # Generate selection rationale
    rationale = generate_model_selection_rationale(comparison_df, best_model_name)
    logger.info("\nModel selection rationale:\n%s", rationale)

    # ------------------------------------------------------------------
    # 8. Generate evaluation plots
    # ------------------------------------------------------------------
    logger.info("\n--- Step 8: Generating evaluation plots ---")
    plot_confusion_matrix(best_pipeline, X_test, y_test, best_model_name)
    plot_roc_curve(best_pipeline, X_test, y_test, best_model_name)

    feature_names = get_feature_names_from_pipeline(best_pipeline)
    plot_feature_importance(best_pipeline, feature_names, best_model_name)
    plot_model_comparison(comparison_df)

    # ------------------------------------------------------------------
    # 9. SHAP explainability
    # ------------------------------------------------------------------
    logger.info("\n--- Step 9: Computing SHAP explanations ---")
    try:
        shap_values, explainer, X_explain = compute_shap_values(
            best_pipeline, X_train, feature_names
        )
        plot_shap_summary(shap_values, X_explain, feature_names)
        top_features = get_top_shap_features(shap_values, feature_names)
        logger.info("\nTop SHAP features:\n%s", top_features.to_string(index=False))
    except Exception as e:
        logger.error("SHAP computation failed: %s. Continuing without SHAP.", e)
        top_features = pd.DataFrame(columns=["Feature", "Mean_SHAP_Value"])

    # ------------------------------------------------------------------
    # 10. Save artefacts
    # ------------------------------------------------------------------
    logger.info("\n--- Step 10: Saving model artefacts ---")

    # Save fitted pipeline
    save_pickle(best_pipeline, MODEL_PATH)
    logger.info("Saved model pipeline to %s", MODEL_PATH)

    # Save preprocessor separately (for potential reuse)
    save_pickle(best_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)
    logger.info("Saved preprocessor to %s", PREPROCESSOR_PATH)

    # Save feature names
    save_json({"feature_names": feature_names}, FEATURE_NAMES_PATH)
    logger.info("Saved feature names to %s", FEATURE_NAMES_PATH)

    # Prepare metrics JSON
    best_test_metrics = comparison_df[comparison_df["model"] == best_model_name].iloc[0].to_dict()
    metrics_data = {
        "best_model": best_model_name,
        "test_metrics": best_test_metrics,
        "cv_results": cv_results,
        "all_models": comparison_df.to_dict(orient="records"),
        "dataset_summary": summary,
        "monthly_charge_median": float(monthly_charge_median),
        "classification_report": report,
        "selection_rationale": rationale,
        "feature_columns": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "top_shap_features": (
            top_features.to_dict(orient="records")
            if not top_features.empty
            else []
        ),
    }
    save_json(metrics_data, METRICS_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("Best model: %s", best_model_name)
    logger.info(
        "F1: %.4f | Recall: %.4f | AUC: %.4f",
        best_test_metrics.get("f1_score", 0),
        best_test_metrics.get("recall", 0),
        best_test_metrics.get("roc_auc", 0),
    )
    logger.info("Artefacts saved in: models/ and outputs/")
    logger.info("Run the app with: streamlit run app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
