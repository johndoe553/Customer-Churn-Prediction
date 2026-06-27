# Explainable Telecom Customer Churn Prediction System

An explainable Streamlit machine-learning application for predicting telecom customer churn using Scikit-learn, XGBoost, SMOTE, and SHAP. It compares candidate classifiers, reports held-out evaluation results, explains predictions, and provides rule-based retention recommendations.

> This system is decision support only. It must not be used for automatic service denial, discrimination, or as a replacement for human review.

## Business problem

Customer churn reduces recurring revenue and increases acquisition costs. This project estimates an individual customer's churn risk from account, service, and billing attributes so retention teams can prioritize human-reviewed interventions and understand the model signals behind them.

## Features

- Validated data cleaning, preprocessing, and feature engineering
- SMOTE applied inside the training pipeline to avoid cross-validation leakage
- Comparison of logistic regression, random forest, gradient boosting, and XGBoost
- Held-out evaluation and stratified cross-validation
- Global and local SHAP explainability
- Rule-based, customer-specific retention recommendations
- Interactive Streamlit pages for prediction, analytics, performance, and responsible use

## Technology

Python, Streamlit, Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, SHAP, Plotly, Matplotlib, and Seaborn.

## Project structure

```text
Customer-Churn-Prediction/
├── app.py                       # Streamlit entry point
├── requirements.txt            # Runtime and training dependencies
├── .streamlit/config.toml       # Cloud-compatible Streamlit configuration
├── data/
│   └── README.md                # Dataset acquisition instructions
├── models/                      # Small deployment artefacts and metadata
├── notebooks/                   # Reproducible experimentation notebook
├── outputs/                     # Selected evaluation tables and figures
├── scripts/
│   ├── train_model.py           # End-to-end model training
│   └── generate_visualisations.py
├── src/                         # Processing, training, evaluation, and explanations
└── tests/                       # Automated tests
```

## Prerequisites

- Python 3.9 or newer
- `pip`
- The dataset below is required for training and dataset analytics. The committed model artefacts are sufficient for the prediction, performance, and explainability pages.

## Dataset

This project uses the **Telco Customer Churn** sample dataset attributed to IBM and distributed on Kaggle by BlastChar. The Kaggle data card marks the files as copyrighted by their original authors and does not provide a clear redistribution licence, so the CSV is intentionally excluded from this repository.

1. Download the dataset from [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Place it at exactly `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

The records describe a fictional telecom company, but users remain responsible for complying with Kaggle's terms and the source dataset's rights. See [data/README.md](data/README.md).

## Installation

Clone the repository and enter it, then create an isolated environment.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train, run, and test

To regenerate all model artefacts and evaluation outputs after adding the dataset:

```bash
python scripts/train_model.py
```

Start the application from the repository root:

```bash
streamlit run app.py
```

Run the automated test suite:

```bash
pip install -r requirements-dev.txt
pytest -v
```

All application paths are derived from the repository root, so no local machine paths or notebook execution are required.

## Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository.
3. Select branch `main` and entry point `app.py`.
4. Deploy without adding secrets; this project does not require any.

The committed model artefacts allow customer prediction without the raw dataset. Dataset Analytics and in-app retraining show a clear message until the CSV is supplied. Community Cloud does not provide a durable mechanism for manually uploading this CSV into a public deployment; if those pages are required, use a data source whose licence explicitly permits redistribution or host the application in an environment where the dataset can be provisioned lawfully.

## Screenshots

Add current screenshots after deployment:

- Dashboard overview — `docs/screenshots/dashboard-overview.png`
- Customer prediction — `docs/screenshots/customer-prediction.png`
- Dataset analytics — `docs/screenshots/dataset-analytics.png`
- Model performance — `docs/screenshots/model-performance.png`
- SHAP explainability — `docs/screenshots/shap-explainability.png`

## Responsible use and limitations

- Predictions are statistical associations, not causal findings or guarantees.
- The sample data may not represent current customers, markets, regions, or protected groups.
- Performance can degrade through data or concept drift; monitor and retrain before operational use.
- SHAP explains model behavior, not the true cause of churn.
- Retention recommendations are rules for human consideration, not automated decisions.
- Review outcomes for disparate impact. Never use predictions to deny service, discriminate, or replace accountable human review.

## Reproducibility and artefacts

Small model artefacts in `models/` and selected evaluation outputs in `outputs/` are versioned to support a working demonstration and academic reproducibility. They contain no credentials or raw customer rows and remain far below GitHub's file-size limits. Regenerate them with `python scripts/train_model.py`; results can vary across dependency versions and platforms.

## Licence

The original project code is licensed under the [MIT License](LICENSE). That licence does **not** apply to the third-party dataset. Dataset use remains subject to the source authors' rights and Kaggle's terms.
