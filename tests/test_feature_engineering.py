import pandas as pd
from src.feature_engineering import engineer_features

def test_engineer_features_tenure_zero():
    # Tenure 0 should be clipped to 1 to avoid division by zero
    df = pd.DataFrame({
        "tenure": [0],
        "TotalCharges": [100.0],
        "MonthlyCharges": [100.0]
    })
    eng = engineer_features(df, monthly_charge_median=50.0)
    assert eng["AverageMonthlySpend"].iloc[0] == 100.0

def test_engineer_features_categories():
    df = pd.DataFrame({
        "tenure": [1, 24, 50],
        "TotalCharges": [10.0, 240.0, 500.0],
        "MonthlyCharges": [10.0, 10.0, 10.0],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"]
    })
    eng = engineer_features(df, monthly_charge_median=50.0)

    assert eng["TenureGroup"].tolist() == ["New", "Early", "Loyal"]
    assert eng["HasSecurityServices"].tolist() == ["Yes", "No", "No"]
    assert eng["HasStreamingServices"].tolist() == ["No", "Yes", "No"]
    assert eng["IsMonthToMonthContract"].tolist() == ["Yes", "No", "No"]
    assert eng["HighMonthlyCharge"].tolist() == ["No", "No", "No"]
    assert eng["FamilySupport"].tolist() == ["Yes", "No", "Yes"]

def test_engineer_features_missing_cols():
    # Should handle missing optional columns safely
    df = pd.DataFrame({
        "tenure": [10],
        "TotalCharges": [100.0],
        "MonthlyCharges": [10.0]
    })
    eng = engineer_features(df, monthly_charge_median=50.0)

    assert eng["HasSecurityServices"].iloc[0] == "No"
    assert eng["HasStreamingServices"].iloc[0] == "No"
    assert eng["IsMonthToMonthContract"].iloc[0] == "No"
    assert eng["FamilySupport"].iloc[0] == "No"
