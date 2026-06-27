"""
model_training.py — Build, compare, and select classification models.

Pipeline architecture:
  - ColumnTransformer for preprocessing (fitted on training data only).
  - SMOTE for class balancing (applied inside imblearn Pipeline, training only).
  - Four classifiers: Logistic Regression, Decision Tree, Random Forest, XGBoost.
  - 5-fold stratified cross-validation on training data.
  - Model selection: F1 (primary) → Recall + AUC (secondary) → interpretability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.utils import get_logger

logger = get_logger(__name__)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
      - Numeric: median imputation + StandardScaler.
      - Categorical: most-frequent imputation + OneHotEncoder.

    Parameters
    ----------
    numeric_features : list of str
    categorical_features : list of str

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_classifiers() -> Dict:
    """
    Return a dictionary of classifier instances.

    Returns
    -------
    dict
        Keys are model names; values are classifier instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,  # SMOTE handles imbalance
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }


def build_pipeline(
    preprocessor: ColumnTransformer,
    classifier,
    apply_smote: bool = True,
) -> ImbPipeline:
    """
    Build an imblearn Pipeline with optional SMOTE.

    Parameters
    ----------
    preprocessor : ColumnTransformer
    classifier : estimator
    apply_smote : bool

    Returns
    -------
    ImbPipeline
    """
    steps = [("preprocessor", preprocessor)]

    if apply_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(("classifier", classifier))

    return ImbPipeline(steps=steps)


def cross_validate_model(
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict:
    """
    Run stratified k-fold cross-validation and return mean/std metrics.

    Parameters
    ----------
    pipeline : ImbPipeline
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    dict
        Keys: cv_f1_mean, cv_f1_std, cv_recall_mean, cv_roc_auc_mean.
    """
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    scoring = {
        "f1": "f1",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
    }

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        "cv_f1_mean": float(np.mean(results["test_f1"])),
        "cv_f1_std": float(np.std(results["test_f1"])),
        "cv_recall_mean": float(np.mean(results["test_recall"])),
        "cv_roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "cv_accuracy_mean": float(np.mean(results["test_accuracy"])),
        "cv_precision_mean": float(np.mean(results["test_precision"])),
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: List[str] = None,
    categorical_features: List[str] = None,
) -> Tuple[Dict, Dict, ColumnTransformer]:
    """
    Train and cross-validate all candidate classifiers.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    numeric_features : list of str, optional
    categorical_features : list of str, optional

    Returns
    -------
    tuple of (trained_pipelines, cv_results, preprocessor)
        - trained_pipelines: {name: fitted ImbPipeline}
        - cv_results: {name: dict of CV metrics}
        - preprocessor: fitted ColumnTransformer (from last pipeline)
    """
    if numeric_features is None:
        numeric_features = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    if categorical_features is None:
        categorical_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    classifiers = get_classifiers()
    trained_pipelines = {}
    cv_results = {}

    for name, clf in classifiers.items():
        logger.info("Training %s ...", name)

        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = build_pipeline(preprocessor, clf, apply_smote=True)

        # Cross-validate
        cv_metrics = cross_validate_model(pipeline, X_train, y_train)
        cv_results[name] = cv_metrics
        logger.info(
            "%s — CV F1: %.4f ± %.4f | Recall: %.4f | AUC: %.4f",
            name,
            cv_metrics["cv_f1_mean"],
            cv_metrics["cv_f1_std"],
            cv_metrics["cv_recall_mean"],
            cv_metrics["cv_roc_auc_mean"],
        )

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

    return trained_pipelines, cv_results, preprocessor


def select_best_model(
    cv_results: Dict,
    interpretability_order: List[str] = None,
) -> str:
    """
    Select the best model using the defined selection rule:
      1. Primary: CV F1-score.
      2. Secondary: Recall + ROC-AUC.
      3. Tiebreak: prefer more interpretable model.

    Parameters
    ----------
    cv_results : dict
        {model_name: {cv_f1_mean, cv_recall_mean, cv_roc_auc_mean, ...}}
    interpretability_order : list of str, optional
        Model names ordered from most to least interpretable.

    Returns
    -------
    str
        Name of the selected model.
    """
    if interpretability_order is None:
        interpretability_order = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
        ]

    # Create composite score: 60% F1 + 20% Recall + 20% AUC
    scores = {}
    for name, metrics in cv_results.items():
        composite = (
            0.60 * metrics["cv_f1_mean"]
            + 0.20 * metrics["cv_recall_mean"]
            + 0.20 * metrics["cv_roc_auc_mean"]
        )
        scores[name] = composite

    # Sort by composite score descending, then by interpretability
    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            interpretability_order.index(item[0])
            if item[0] in interpretability_order
            else 999,
        ),
    )

    best_name = ranked[0][0]

    # Check if top two are within 0.02 — prefer more interpretable
    if len(ranked) >= 2:
        top_score = ranked[0][1]
        second_name, second_score = ranked[1]
        if abs(top_score - second_score) < 0.02:
            top_idx = (
                interpretability_order.index(ranked[0][0])
                if ranked[0][0] in interpretability_order
                else 999
            )
            second_idx = (
                interpretability_order.index(second_name)
                if second_name in interpretability_order
                else 999
            )
            if second_idx < top_idx:
                best_name = second_name
                logger.info(
                    "Performance difference < 0.02; preferring more interpretable "
                    "model: %s",
                    best_name,
                )

    logger.info("Selected best model: %s (composite: %.4f)", best_name, scores[best_name])
    return best_name


def get_feature_names_from_pipeline(pipeline: ImbPipeline) -> List[str]:
    """
    Extract final feature names after preprocessing (OneHotEncoding).

    Parameters
    ----------
    pipeline : ImbPipeline

    Returns
    -------
    list of str
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps["encoder"]
            encoded_names = encoder.get_feature_names_out(columns).tolist()
            feature_names.extend(encoded_names)

    return feature_names
