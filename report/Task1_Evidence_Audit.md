# Task 1 Evidence Audit

This document summarises the verified evidence extracted from the local project codebase. All findings are derived directly from the code, generated artefacts, and log outputs.

## 1. Files Inspected
- `src/data_processing.py`
- `src/feature_engineering.py`
- `src/model_training.py`
- `outputs/model_comparison.csv`
- `README.md`
- `.gitignore`
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Existence verified)

## 2. Verified Dataset Facts
- **Source**: IBM Telco Customer Churn dataset distributed via Kaggle.
- **Dataset location**: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Size: 948 KB)
- **Target Variable**: `Churn` (encoded as binary: 1 = "Yes", 0 = "No")
- **Row count**: 7043
- **Data Cleaning Steps**:
  - Duplicate rows removed.
  - Object columns: Whitespace stripped.
  - `SeniorCitizen` mapped from 0/1 to "No"/"Yes".
  - `TotalCharges`: Blanks and whitespace-only strings replaced with NaN, then imputed with the median (1397.47).
  - Categorical variables imputed with the mode; numeric variables imputed with the median.
  - `CustomerID` explicitly dropped.

## 3. Verified Features and Engineering
**Engineered Features (7)**:
1. `TenureGroup`: Binned `tenure` into "New" (0-12), "Early" (13-24), "Established" (25-48), "Loyal" (>48).
2. `AverageMonthlySpend`: `TotalCharges` / `tenure` (clipped at lower bound of 1).
3. `HasSecurityServices`: "Yes" if `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, or `TechSupport` is "Yes".
4. `HasStreamingServices`: "Yes" if `StreamingTV` or `StreamingMovies` is "Yes".
5. `IsMonthToMonthContract`: "Yes" if `Contract` == "Month-to-month".
6. `HighMonthlyCharge`: "Yes" if `MonthlyCharges` >= dataset median (fallback default 70.35).
7. `FamilySupport`: "Yes" if `Partner` == "Yes" or `Dependents` == "Yes".

## 4. Verified Preprocessing and Pipeline
- **Numeric Features**: Median Imputation + `StandardScaler`
- **Categorical Features**: Mode Imputation + `OneHotEncoder` (handle_unknown='ignore')
- **Class Balancing**: `SMOTE` applied via `imblearn.pipeline` (applied *only* to the training data inside cross-validation loops to prevent target leakage).
- **Cross-Validation**: 5-fold Stratified K-Fold.

## 5. Verified Models and Hyperparameters
1. **Logistic Regression**: `max_iter=1000`, `class_weight="balanced"`, `solver="lbfgs"`
2. **Decision Tree**: `max_depth=8`, `class_weight="balanced"`
3. **Random Forest**: `n_estimators=200`, `max_depth=12`, `class_weight="balanced"`
4. **XGBoost**: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric="logloss"`, `use_label_encoder=False`

## 6. Verified Evaluation Metrics (from `outputs/model_comparison.csv`)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV F1 Mean | CV F1 Std |
|-------|----------|-----------|--------|----------|---------|------------|-----------|
| **Logistic Regression** | 0.7417 | 0.5086 | **0.7888** | **0.6184** | **0.8396** | 0.6299 | 0.0253 |
| **Decision Tree** | 0.7615 | 0.5411 | 0.6684 | 0.5981 | 0.8025 | 0.5868 | 0.0214 |
| **Random Forest** | 0.7757 | 0.5668 | 0.6578 | 0.6089 | 0.8337 | 0.6182 | 0.0179 |
| **XGBoost** | **0.7771** | **0.5820** | 0.5695 | 0.5757 | 0.8321 | 0.5939 | 0.0327 |

*Note: Logistic Regression was selected as the final model based on the composite rule scoring highest, prioritizing recall for the minority churn class.*

## 7. Unknown/Required Elements for the Final Report
- Student Name / Student ID.
- Exact screenshots to be taken by the user.

## 8. Deployment Details
- **GitHub URL**: `https://github.com/johndoe553/Customer-Churn-Prediction`
- **Streamlit URL**: `https://explainable-telecom-customer-churn-prediction.streamlit.app/`
