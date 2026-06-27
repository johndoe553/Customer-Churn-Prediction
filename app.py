"""
app.py — Explainable Telecom Customer Churn Prediction Dashboard.

A production-quality Streamlit application with six pages:
  1. Home / Project Overview
  2. Customer Churn Prediction
  3. Dataset Analytics
  4. Model Performance
  5. Explainability / Model Insights
  6. Responsible Use & Limitations

Launch with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_NAMES_PATH,
    SHAP_BACKGROUND_PATH,
    DATASET_PATH,
    CONFUSION_MATRIX_PNG,
    ROC_CURVE_PNG,
    FEATURE_IMPORTANCE_PNG,
    SHAP_SUMMARY_PNG,
    CHURN_DISTRIBUTION_PNG,
    CONTRACT_CHURN_PNG,
    TENURE_CHURN_PNG,
    MONTHLY_CHARGES_CHURN_PNG,
    MODEL_COMPARISON_CSV,
    OUTPUTS_DIR,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    COLOUR_PALETTE,
)
from src.utils import load_pickle, load_json
from src.feature_engineering import engineer_features
from src.retention_recommendations import (
    generate_recommendations,
    format_risk_badge,
    get_risk_level,
)
from src.explainability import explain_single_prediction

# =========================================================================
# Page configuration
# =========================================================================
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# Custom CSS
# =========================================================================
st.markdown(
    """
    <style>
    /* Main container spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #0068C9;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        margin: 8px 0;
        color: #FFFFFF;
    }
    .risk-high   { background-color: #E74C3C; }
    .risk-medium { background-color: #F39C12; }
    .risk-low    { background-color: #27AE60; }

    /* Recommendation cards */
    .rec-card {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-left: 4px solid #0068C9;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #F0F4F8; }

    /* Section divider */
    .section-divider {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }

    /* Responsible use warning */
    .responsible-use {
        background: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Cached loaders
# =========================================================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    return load_pickle(MODEL_PATH)


@st.cache_resource
def load_background_data():
    """Load SHAP background data."""
    return load_pickle(SHAP_BACKGROUND_PATH)


@st.cache_data
def load_metrics():
    """Load model metrics JSON."""
    return load_json(METRICS_PATH)


@st.cache_data
def load_feature_names():
    """Load feature names list."""
    data = load_json(FEATURE_NAMES_PATH)
    return data.get("feature_names", [])


@st.cache_data
def load_dataset_cached():
    """Load and clean the dataset for analytics."""
    from src.data_processing import load_dataset, clean_dataset
    df = load_dataset()
    return clean_dataset(df)


@st.cache_data
def load_comparison_csv():
    """Load model comparison CSV."""
    return pd.read_csv(MODEL_COMPARISON_CSV)


def check_artefacts_exist() -> bool:
    """Check if required model artefacts exist."""
    return MODEL_PATH.exists() and METRICS_PATH.exists()


# =========================================================================
# Sidebar navigation
# =========================================================================
st.sidebar.markdown("## 📡 Churn Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Churn Prediction",
        "📊 Dataset Analytics",
        "📈 Model Performance",
        "🔍 Explainability",
        "⚙️ Model Training",
        "⚖️ Responsible Use",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Telecom Customer Churn Prediction System  \n"
    "Built with Scikit-learn, XGBoost, SHAP & Streamlit"
)


# =========================================================================
# PAGE 1: Home / Project Overview
# =========================================================================
if page == "🏠 Home":
    st.title("📡 Explainable Telecom Customer Churn Prediction")
    st.markdown(
        "#### Identify at-risk customers early and take data-driven retention actions."
    )

    st.markdown("---")

    st.markdown(
        """
        **Business Problem:** Telecom companies lose significant revenue from customer
        churn. This system uses machine learning to predict which customers are likely
        to leave, enabling retention teams to intervene proactively with personalised
        offers and support.
        """
    )

    # Dataset summary metrics
    if check_artefacts_exist():
        try:
            metrics = load_metrics()
            summary = metrics.get("dataset_summary", {})
            best_model = metrics.get("best_model", "N/A")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", f"{summary.get('n_customers', 0):,}")
            with col2:
                st.metric("Input Features", summary.get("n_features", 0))
            with col3:
                churn_rate = summary.get("churn_rate", 0)
                st.metric("Churn Rate", f"{churn_rate * 100:.1f}%")
            with col4:
                st.metric("Selected Model", best_model)

            st.markdown("---")

            # Quick overview cards
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### 📋 How It Works")
                st.markdown(
                    """
                    1. **Data Collection** — Historical customer data is analysed.
                    2. **Feature Engineering** — Key signals are extracted (tenure,
                       contract type, services, charges).
                    3. **ML Prediction** — A trained classifier estimates churn
                       probability.
                    4. **Explainability** — SHAP values reveal why the model made
                       its prediction.
                    5. **Retention Actions** — Recommendations guide the retention
                       team's response.
                    """
                )
            with col_b:
                st.markdown("##### 🎯 Key Capabilities")
                st.markdown(
                    """
                    - Predict individual customer churn risk.
                    - Understand top churn drivers using SHAP.
                    - Receive actionable retention recommendations.
                    - Explore dataset patterns and model performance.
                    - Support transparent, human-reviewed decisions.
                    """
                )
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    else:
        st.warning(
            "⚠️ **Model artefacts not found.** Please train the model first:\n\n"
            "```bash\npython scripts/train_model.py\n```"
        )

    # Responsible use warning
    st.markdown("---")
    st.markdown(
        '<div class="responsible-use">'
        "⚠️ <strong>Responsible Use Notice:</strong> This tool provides "
        "decision-support predictions only. It should not automatically deny "
        "service, target customers unfairly, or replace human review. "
        "Predictions should be used alongside professional judgement to guide "
        "customer-retention strategies."
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# PAGE 2: Customer Churn Prediction
# =========================================================================
elif page == "🔮 Churn Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Enter customer details to predict their churn risk and receive tailored retention recommendations.")

    if not check_artefacts_exist():
        st.error(
            "❌ Model artefacts not found. Please run:\n\n"
            "```bash\npython scripts/train_model.py\n```"
        )
        st.stop()

    try:
        model = load_model()
        metrics = load_metrics()
        feature_names = load_feature_names()
        monthly_charge_median = metrics.get("monthly_charge_median", 70.35)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # ----- Input form -----
    with st.form("prediction_form"):
        st.markdown("### A. Customer Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        with col3:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
                help="How long the customer has been with the company (0–72 months).",
            )

        st.markdown("---")
        st.markdown("### B. Services")
        col4, col5, col6 = st.columns(3)
        with col4:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )
        with col5:
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
            )
        with col6:
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
            )

        st.markdown("---")
        st.markdown("### C. Account & Billing")
        col7, col8, col9 = st.columns(3)
        with col7:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with col8:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with col9:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=500.0,
                value=70.35,
                step=0.01,
                format="%.2f",
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=50000.0,
                value=float(monthly_charges * max(tenure, 1)),
                step=0.01,
                format="%.2f",
                help="Cumulative charges over the customer's tenure.",
            )

        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk", width="stretch"
        )

    # ----- Prediction -----
    if submitted:
        # Validate inputs
        errors = []
        if total_charges < 0:
            errors.append("Total Charges cannot be negative.")
        if not (0 <= tenure <= 72):
            errors.append("Tenure must be between 0 and 72 months.")

        if errors:
            for err in errors:
                st.error(err)
            st.stop()

        # Build input dataframe
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([input_data])

        # Apply same feature engineering as training
        input_df = engineer_features(
            input_df, monthly_charge_median=monthly_charge_median
        )

        with st.spinner("Analysing customer profile..."):
            try:
                probability = model.predict_proba(input_df)[:, 1][0]
                prediction = model.predict(input_df)[0]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # ----- Results -----
        st.markdown("---")
        st.markdown("## Prediction Results")

        badge = format_risk_badge(probability)
        risk_level = badge["level"]
        risk_class = f"risk-{risk_level.lower()}"

        col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
        with col_r1:
            st.metric("Churn Probability", f"{probability * 100:.1f}%")
        with col_r2:
            st.markdown(
                f'<div class="risk-badge {risk_class}">'
                f'{badge["icon"]} {badge["label"]}'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_r3:
            st.metric(
                "Risk Level",
                risk_level,
                delta=f"{probability * 100:.1f}% probability",
                delta_color="inverse",
            )

        # Retention recommendations
        st.markdown("### 💡 Retention Recommendations")
        st.caption(
            "These are decision-support suggestions, not guaranteed outcomes. "
            "Always apply professional judgement."
        )

        recommendations = generate_recommendations(probability, input_data)
        for rec in recommendations:
            st.markdown(
                f'<div class="rec-card">'
                f'<strong>{rec["category"]}</strong><br>'
                f'{rec["recommendation"]}<br>'
                f'<em style="color: #64748B; font-size: 0.85rem;">'
                f'Rationale: {rec["rationale"]}</em>'
                f"</div>",
                unsafe_allow_html=True,
            )

        # Local SHAP explanation
        st.markdown("### 🔍 Key Risk Contributors")
        try:
            background = load_background_data()
            explanation = explain_single_prediction(
                model, input_df, feature_names, background
            )
            if explanation is not None:
                sv = explanation["shap_values"]
                fn = explanation["feature_names"]
                n = min(len(sv), len(fn))

                shap_df = pd.DataFrame({
                    "Feature": fn[:n],
                    "Impact": sv[:n],
                }).sort_values("Impact", key=abs, ascending=False).head(10)

                fig_local = go.Figure()
                colours = [
                    COLOUR_PALETTE["danger"] if v > 0 else COLOUR_PALETTE["success"]
                    for v in shap_df["Impact"]
                ]
                fig_local.add_trace(
                    go.Bar(
                        y=shap_df["Feature"],
                        x=shap_df["Impact"],
                        orientation="h",
                        marker_color=colours,
                    )
                )
                fig_local.update_layout(
                    title="Feature Contributions to This Prediction",
                    xaxis_title="SHAP Value (impact on churn prediction)",
                    yaxis=dict(autorange="reversed"),
                    height=400,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_local, width="stretch")
                st.caption(
                    "🔴 Red bars push toward churn · 🟢 Green bars push away from churn. "
                    "These are model associations, not proof of causation."
                )
            else:
                st.info("Local SHAP explanations are unavailable for this model configuration.")
        except FileNotFoundError:
            st.info(
                "SHAP background data not found. Run training to enable local explanations."
            )
        except Exception as e:
            st.warning(f"Could not generate local explanation: {e}")


# =========================================================================
# PAGE 3: Dataset Analytics
# =========================================================================
elif page == "📊 Dataset Analytics":
    st.title("📊 Dataset Analytics")
    st.markdown("Explore patterns in the Telco Customer Churn dataset.")

    if not DATASET_PATH.exists():
        st.warning(
            "⚠️ Dataset not found. Please place `WA_Fn-UseC_-Telco-Customer-Churn.csv` "
            "in the `data/` directory."
        )
        st.stop()

    try:
        df = load_dataset_cached()
        df_eng = engineer_features(df)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_count = (df["Churn"] == "Yes").sum()
        st.metric("Churned Customers", f"{churn_count:,}")
    with col3:
        churn_rate = churn_count / len(df) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        avg_tenure = df["tenure"].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

    st.markdown("---")

    # Tab layout for charts
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Churn Distribution",
        "Contract Type",
        "Tenure Groups",
        "Monthly Charges",
        "Correlations",
    ])

    with tab1:
        st.markdown("#### Customer Churn Distribution")
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        fig1 = px.bar(
            counts,
            x="Churn",
            y="Count",
            color="Churn",
            color_discrete_map={"No": COLOUR_PALETTE["primary"], "Yes": COLOUR_PALETTE["danger"]},
            text="Count",
            template="plotly_white",
        )
        fig1.update_layout(
            showlegend=False,
            xaxis_title="Churn Status",
            yaxis_title="Number of Customers",
            height=400,
        )
        fig1.update_traces(textposition="outside")
        st.plotly_chart(fig1, width="stretch")
        st.info(
            f"💡 **Insight:** {churn_rate:.1f}% of customers churned, making this an "
            f"imbalanced classification problem. Accuracy alone can be misleading — "
            f"a model predicting 'No Churn' for everyone would achieve "
            f"~{100 - churn_rate:.0f}% accuracy but miss all churners."
        )

    with tab2:
        st.markdown("#### Churn Rate by Contract Type")
        ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index").reset_index()
        ct_melted = ct.melt(id_vars="Contract", var_name="Churn", value_name="Proportion")
        ct_melted["Percentage"] = ct_melted["Proportion"] * 100

        fig2 = px.bar(
            ct_melted,
            x="Contract",
            y="Percentage",
            color="Churn",
            barmode="group",
            color_discrete_map={"No": COLOUR_PALETTE["primary"], "Yes": COLOUR_PALETTE["danger"]},
            text=ct_melted["Percentage"].round(1),
            template="plotly_white",
        )
        fig2.update_layout(
            yaxis_title="Percentage (%)",
            xaxis_title="Contract Type",
            height=400,
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, width="stretch")
        st.info(
            "💡 **Insight:** Month-to-month contracts show the highest churn rate. "
            "Customers on longer-term contracts have significantly lower churn, "
            "suggesting that commitment reduces attrition."
        )

    with tab3:
        st.markdown("#### Churn Rate by Tenure Group")
        order = ["New", "Early", "Established", "Loyal"]
        ct_tenure = pd.crosstab(df_eng["TenureGroup"], df_eng["Churn"], normalize="index").reset_index()
        ct_tenure_melted = ct_tenure.melt(id_vars="TenureGroup", var_name="Churn", value_name="Proportion")
        ct_tenure_melted["Percentage"] = ct_tenure_melted["Proportion"] * 100
        ct_tenure_melted["TenureGroup"] = pd.Categorical(
            ct_tenure_melted["TenureGroup"], categories=order, ordered=True
        )
        ct_tenure_melted = ct_tenure_melted.sort_values("TenureGroup")

        fig3 = px.bar(
            ct_tenure_melted,
            x="TenureGroup",
            y="Percentage",
            color="Churn",
            barmode="group",
            color_discrete_map={"No": COLOUR_PALETTE["primary"], "Yes": COLOUR_PALETTE["danger"]},
            text=ct_tenure_melted["Percentage"].round(1),
            template="plotly_white",
            category_orders={"TenureGroup": order},
        )
        fig3.update_layout(
            yaxis_title="Percentage (%)",
            xaxis_title="Tenure Group",
            height=400,
        )
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, width="stretch")
        st.info(
            "💡 **Insight:** New customers (0–12 months) have the highest churn rate. "
            "Churn decreases substantially with tenure, underscoring the importance "
            "of early-life engagement and onboarding programmes."
        )

    with tab4:
        st.markdown("#### Monthly Charges Distribution by Churn Status")
        fig4 = px.histogram(
            df,
            x="MonthlyCharges",
            color="Churn",
            barmode="overlay",
            nbins=40,
            color_discrete_map={"No": COLOUR_PALETTE["primary"], "Yes": COLOUR_PALETTE["danger"]},
            opacity=0.7,
            template="plotly_white",
        )
        fig4.update_layout(
            xaxis_title="Monthly Charges ($)",
            yaxis_title="Number of Customers",
            height=400,
        )
        st.plotly_chart(fig4, width="stretch")
        st.info(
            "💡 **Insight:** Churned customers tend to have higher monthly charges. "
            "This may indicate price sensitivity or misalignment between plan cost "
            "and perceived value."
        )

    with tab5:
        st.markdown("#### Correlation Heatmap (Numerical Features)")
        numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
        numeric_df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
        corr = numeric_df.corr()

        fig5 = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_white",
        )
        fig5.update_layout(
            title="Feature Correlations",
            height=450,
        )
        st.plotly_chart(fig5, width="stretch")
        st.info(
            "💡 **Insight:** Tenure and TotalCharges are positively correlated "
            "(long-tenured customers accumulate more charges). Monthly charges "
            "show a moderate positive correlation with churn."
        )


# =========================================================================
# PAGE 4: Model Performance
# =========================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Evaluate and compare model classification performance.")

    if not check_artefacts_exist():
        st.warning(
            "⚠️ Model artefacts not found. Please train the model first:\n\n"
            "```bash\npython scripts/train_model.py\n```"
        )
        st.stop()

    try:
        metrics = load_metrics()
        best_model = metrics.get("best_model", "N/A")
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.stop()

    # Selected model banner
    st.success(f"✅ **Selected Model: {best_model}**")

    # Model comparison table
    st.markdown("### 📋 Model Comparison")
    all_models = metrics.get("all_models", [])
    if all_models:
        comp_df = pd.DataFrame(all_models)
        display_cols = [
            "model", "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "cv_f1_mean", "cv_f1_std",
        ]
        available_cols = [c for c in display_cols if c in comp_df.columns]
        comp_display = comp_df[available_cols].copy()

        # Rename for readability
        rename_map = {
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "roc_auc": "ROC-AUC",
            "cv_f1_mean": "CV F1 Mean",
            "cv_f1_std": "CV F1 Std",
        }
        comp_display = comp_display.rename(columns=rename_map)

        # Highlight best model row
        def highlight_best(row):
            if row["Model"] == best_model:
                return ["background-color: #E8F5E9"] * len(row)
            return [""] * len(row)

        st.dataframe(
            comp_display.style.apply(highlight_best, axis=1).format(
                {c: "{:.4f}" for c in comp_display.columns if c != "Model"},
                na_rep="-",
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Model comparison data not available.")

    st.markdown("---")

    # Model selection rationale
    st.markdown("### 🎯 Model Selection Rationale")
    rationale = metrics.get("selection_rationale", "")
    if rationale:
        st.markdown(rationale)
    else:
        st.info("Selection rationale not available.")

    # Classification report
    st.markdown("### 📄 Classification Report")
    report = metrics.get("classification_report", "")
    if report:
        st.code(report)
        st.caption(
            "**Reading the report:** *Precision* = of all predicted churners, how many "
            "actually churned. *Recall* = of all actual churners, how many did the "
            "model catch. *F1-Score* = harmonic mean of precision and recall."
        )

    st.markdown("---")

    # Evaluation plots
    st.markdown("### 📊 Evaluation Visualisations")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        if CONFUSION_MATRIX_PNG.exists():
            st.image(str(CONFUSION_MATRIX_PNG), width="stretch")
            st.caption(
                "The confusion matrix shows prediction outcomes. Top-left: correctly "
                "predicted non-churners. Bottom-right: correctly predicted churners. "
                "Off-diagonal cells are errors."
            )
        else:
            st.info("Confusion matrix not generated yet.")

    with col_roc:
        st.markdown("#### ROC Curve")
        if ROC_CURVE_PNG.exists():
            st.image(str(ROC_CURVE_PNG), width="stretch")
            st.caption(
                "The ROC curve plots the trade-off between catching churners (true "
                "positive rate) and falsely flagging non-churners (false positive rate). "
                "AUC closer to 1.0 indicates better discrimination."
            )
        else:
            st.info("ROC curve not generated yet.")

    # Feature importance
    if FEATURE_IMPORTANCE_PNG.exists():
        st.markdown("#### Feature Importance")
        st.image(str(FEATURE_IMPORTANCE_PNG), width="stretch")
        st.caption(
            "Feature importances show which variables the model relied on most. "
            "Higher values indicate stronger influence on predictions."
        )

    # Model comparison chart
    model_comp_chart = OUTPUTS_DIR / "model_comparison_chart.png"
    if model_comp_chart.exists():
        st.markdown("#### Model Comparison Chart")
        st.image(str(model_comp_chart), width="stretch")

    # Metric interpretation
    st.markdown("---")
    st.markdown("### 📖 Understanding the Metrics")
    with st.expander("Click to expand metric definitions"):
        st.markdown(
            """
            | Metric | What It Measures | Business Meaning |
            |--------|------------------|------------------|
            | **Accuracy** | Overall correct predictions | Can be misleading when churn is rare |
            | **Precision** | Of predicted churners, how many really churned | Reduces wasted retention spend on wrong customers |
            | **Recall** | Of actual churners, how many did we catch | Ensures we don't miss at-risk customers |
            | **F1-Score** | Balance of precision and recall | Best single metric for imbalanced problems |
            | **ROC-AUC** | Model's ability to rank churners higher | Shows discrimination ability across all thresholds |
            | **CV F1** | Cross-validated F1 on training data | Estimates how well the model generalises |
            """
        )
        st.markdown(
            "⚠️ **Why not just use accuracy?** In this dataset, ~73% of customers "
            "did not churn. A model that always predicts 'No Churn' would be ~73% "
            "accurate — but completely useless for identifying at-risk customers. "
            "F1-score, which balances precision and recall, is a more appropriate "
            "primary metric for this imbalanced problem."
        )


# =========================================================================
# PAGE 5: Explainability / Model Insights
# =========================================================================
elif page == "🔍 Explainability":
    st.title("🔍 Model Explainability & Insights")
    st.markdown(
        "Understand **what drives the model's predictions** using SHAP "
        "(SHapley Additive exPlanations) values."
    )

    if not check_artefacts_exist():
        st.warning(
            "⚠️ Model artefacts not found. Please train the model first:\n\n"
            "```bash\npython scripts/train_model.py\n```"
        )
        st.stop()

    try:
        metrics = load_metrics()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        st.stop()

    # Global SHAP summary
    st.markdown("### 🌐 Global Feature Impact (SHAP Summary)")
    if SHAP_SUMMARY_PNG.exists():
        st.image(str(SHAP_SUMMARY_PNG), width="stretch")
        st.caption(
            "Each dot represents a customer. Colour indicates feature value "
            "(red = high, blue = low). Position on the x-axis shows the feature's "
            "impact on the churn prediction. Features are ordered by overall importance."
        )
    else:
        st.info(
            "SHAP summary plot not generated. This may occur if SHAP computation "
            "was skipped during training."
        )

    st.markdown("---")

    # Top churn drivers
    st.markdown("### 🏆 Top Churn Drivers")
    top_features = metrics.get("top_shap_features", [])
    if top_features:
        top_df = pd.DataFrame(top_features)
        fig_top = px.bar(
            top_df.head(10),
            x="Mean_SHAP_Value",
            y="Feature",
            orientation="h",
            template="plotly_white",
            color="Mean_SHAP_Value",
            color_continuous_scale=["#83C9FF", "#0068C9"],
        )
        fig_top.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="",
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_top, width="stretch")
    else:
        st.info("Top SHAP feature data not available.")

    st.markdown("---")

    # Plain-English interpretations
    st.markdown("### 📝 What Do These Features Mean?")
    st.markdown(
        "*The following interpretations are based on patterns the model learned "
        "from historical data. They represent **associations**, not proof of "
        "causation.*"
    )

    interpretations = [
        (
            "📋 Contract Type",
            "Month-to-month contracts were associated with higher churn risk in this "
            "dataset. Customers without long-term commitments have lower switching "
            "costs and may leave more easily."
        ),
        (
            "⏱️ Customer Tenure",
            "Newer customers tend to churn more than long-standing ones. The first "
            "12 months appear to be a critical period for customer retention."
        ),
        (
            "💰 Monthly Charges",
            "Higher monthly charges were linked to increased churn. Customers paying "
            "more may have higher expectations or feel they are not receiving "
            "proportional value."
        ),
        (
            "🛠️ Technical Support",
            "Customers without tech support were more likely to churn. Access to "
            "support may help resolve issues that would otherwise drive customers away."
        ),
        (
            "🔒 Online Security",
            "Customers without online security services showed higher churn risk. "
            "Additional services may increase perceived value and create switching "
            "barriers."
        ),
        (
            "📡 Internet Service Type",
            "Fibre optic customers showed higher churn rates, possibly due to higher "
            "costs or elevated service quality expectations."
        ),
        (
            "💳 Payment Method",
            "Electronic cheque payment users showed higher churn, potentially "
            "indicating less engagement with automatic payment options."
        ),
    ]

    for title, explanation in interpretations:
        with st.expander(title):
            st.markdown(explanation)

    st.markdown("---")
    st.info(
        "⚠️ **Important:** These are model-learned associations from historical data, "
        "**not proof of causation**. Correlation does not imply that changing a single "
        "feature will directly reduce churn. Business decisions should consider domain "
        "expertise and customer context alongside these insights."
    )


# =========================================================================
# PAGE 6: Model Training
# =========================================================================
elif page == "⚙️ Model Training":
    st.title("⚙️ Model Training")
    st.markdown(
        "Train the predictive models on the dataset directly from the browser. "
        "This will execute the data processing, feature engineering, cross-validation, "
        "and SHAP explanations pipeline, and save the updated model artefacts."
    )

    st.info("💡 **Note:** Training may take 1–3 minutes depending on your system.")

    if st.button("🚀 Start Model Training", type="primary"):
        with st.spinner("Training models in the background... Please wait."):
            import subprocess
            import sys

            try:
                # Run the training script as a subprocess
                result = subprocess.run(
                    [sys.executable, "scripts/train_model.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                st.success("✅ Model training completed successfully!")

                st.markdown(
                    "**The new model has been saved and is ready to use!**\\n\\n"
                    "Head over to the **📈 Model Performance** page to view the latest evaluation metrics, "
                    "or the **🔮 Churn Prediction** page to try it out on new data."
                )

                st.markdown("### 📜 Training Logs")
                with st.expander("View detailed execution logs", expanded=True):
                    st.code(result.stdout, language="text")

                # Clear Streamlit caches so new artefacts are loaded on the next page view
                st.cache_resource.clear()
                st.cache_data.clear()

            except subprocess.CalledProcessError as e:
                st.error("❌ An error occurred during model training.")
                st.markdown("### Error Output")
                with st.expander("View error logs", expanded=True):
                    st.code(e.stderr, language="text")

# =========================================================================
# PAGE 7: Responsible Use & Limitations
# =========================================================================
elif page == "⚖️ Responsible Use":
    st.title("⚖️ Responsible Use & Limitations")
    st.markdown(
        "Understanding the boundaries and ethical considerations of this "
        "prediction system is essential for appropriate deployment."
    )

    st.markdown("---")

    st.markdown("### 📌 Dataset Limitations")
    st.markdown(
        """
        - **Not universally representative:** The training dataset comes from a
          single telecom provider. Patterns may not generalise to other companies,
          markets, or geographic regions.
        - **Historical snapshot:** The data reflects customer behaviour at a
          specific point in time. Market conditions, pricing, and customer
          expectations change over time.
        - **Potential bias:** Historical customer data can contain biases related
          to demographics, geography, or socioeconomic factors that may be
          reflected in the model's predictions.
        """
    )

    st.markdown("### 🔄 Model Drift & Maintenance")
    st.markdown(
        """
        - **Predictions may degrade:** As customer behaviour, market conditions,
          and company offerings evolve, the model's accuracy may decrease.
        - **Regular retraining:** The model should be retrained periodically
          (e.g., quarterly) with fresh data to maintain relevance.
        - **Monitor performance:** Track prediction accuracy over time using
          actual churn outcomes to detect performance degradation.
        - **Feature relevance:** New services, pricing structures, or external
          factors may require updating the feature set.
        """
    )

    st.markdown("### 👥 Human-in-the-Loop")
    st.markdown(
        """
        - **Decision support, not automation:** Predictions should support
          human customer-retention decisions, not fully automate them.
        - **Professional judgement:** Retention strategies should consider
          individual customer context, relationship history, and business
          judgement — not just a probability score.
        - **Escalation paths:** High-impact decisions (e.g., large discounts
          or contract modifications) should include human review and approval.
        """
    )

    st.markdown("### 🚫 Prohibited Uses")
    st.markdown(
        """
        - **Do not discriminate:** Never use predictions to discriminate against
          protected groups (based on gender, age, ethnicity, or other protected
          characteristics) or deny service unfairly.
        - **Do not auto-terminate:** Never automatically terminate or degrade
          service based solely on model predictions.
        - **Do not penalise:** Predictions should drive positive retention
          actions (incentives, support, engagement) — not penalties or
          restrictions.
        - **Privacy:** Customer data used for predictions must comply with
          applicable data protection regulations (e.g., GDPR, CCPA).
        """
    )

    st.markdown("### ✅ Best Practices for Deployment")
    st.markdown(
        """
        1. **Validate before deploying** — Test model performance on recent,
           representative data before live deployment.
        2. **A/B test retention actions** — Measure whether recommended actions
           actually reduce churn before scaling them.
        3. **Document decisions** — Record how predictions influenced retention
           strategies for accountability and audit trails.
        4. **Gather feedback** — Collect feedback from retention teams on
           prediction quality and recommendation usefulness.
        5. **Maintain transparency** — Be prepared to explain to customers how
           their data is used, in compliance with privacy regulations.
        """
    )

    st.markdown("---")
    st.markdown(
        '<div class="responsible-use">'
        "⚠️ <strong>Summary:</strong> This tool is designed to help customer-retention "
        "teams identify at-risk customers and take proactive, supportive action. "
        "It is a decision-support tool — not an automated decision-maker. "
        "Always apply professional judgement, respect customer privacy, and comply "
        "with applicable regulations."
        "</div>",
        unsafe_allow_html=True,
    )
