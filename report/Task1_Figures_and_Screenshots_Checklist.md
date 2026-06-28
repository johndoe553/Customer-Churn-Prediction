# Figures and Screenshots Checklist

Use this checklist to correctly embed the required figures into your final Word document submission.

| Fig. | Title | Status | Source Location / Instruction | Insertion Point |
|---|---|---|---|---|
| 1 | End-to-End Customer Churn Prediction Workflow | [ ] | You will need to draw a workflow diagram (e.g., using draw.io or PowerPoint) showing: Dataset → cleaning → feature engineering → preprocessing → SMOTE → model training → evaluation → SHAP → Streamlit deployment. | Section 1, after paragraph on project aim. |
| 2 | Dataset and Feature Summary (Table 1) | [ ] | N/A (Already formatted as a markdown table in the report) | Section 2, after introduction to dataset. |
| 3 | Churn Distribution Before Resampling | [ ] | Generate locally or run `python scripts/generate_visualisations.py` if present, or take a screenshot from the Streamlit "Dataset Analytics" page showing the churn vs non-churn count. | Section 2, after discussion of the target variable. |
| 4 | Preprocessing and Feature-Engineering Code | [ ] | Take a screenshot of lines 59-109 in `src/feature_engineering.py`. Focus specifically on the engineered features logic (e.g., `TenureGroup` and `AverageMonthlySpend`). | Section 2, after engineered features discussion. |
| 5 | Class Balance After SMOTE | [ ] | Screenshot the terminal output of the training script (`python scripts/train_model.py`) showing the SMOTE application step and the resulting balanced class distribution inside the pipeline. | Section 2, after SMOTE justification. |
| 6 | Model Training and Cross-Validation Code | [ ] | Take a screenshot of lines 150-198 in `src/model_training.py` showing the `cross_validate_model` function and `StratifiedKFold` setup. | Section 3, after cross-validation discussion. |
| 7 | Model Comparison Results (Table 2) | [ ] | N/A (Already populated with actual CSV data in the report) | Section 4, after model comparison introduction. |
| 8 | Final Model Confusion Matrix | [ ] | Insert the image generated at `outputs/confusion_matrix.png`. | Section 4, after confusion matrix discussion. |
| 9 | Final Model ROC Curve | [ ] | Insert the image generated at `outputs/roc_curve.png`. | Section 4, after ROC curve discussion. |
| 10 | SHAP Global Feature Importance | [ ] | Insert the image generated at `outputs/shap_summary.png`. | Section 4, after SHAP discussion. |
| 11 | Streamlit Customer Prediction Page | [ ] | Run the app (`streamlit run app.py`), go to the prediction page, fill out a sample profile, and screenshot the risk gauge and result. | Section 5, after Streamlit capabilities discussion. |
| 12 | Streamlit Explainability Page | [ ] | Navigate to the "Explainability" tab in the deployed or local Streamlit app and capture a screenshot of the local SHAP waterfall chart or the global summary. | Section 5, after mentioning the explainability view. |
