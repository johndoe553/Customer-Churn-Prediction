import pandas as pd
import numpy as np
import pytest
from src.data_processing import clean_dataset, encode_target

def test_clean_dataset_total_charges_blank():
    # Test blank TotalCharges becomes NaN and is imputed
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "TotalCharges": ["10.5", " ", "", "20.0"],
        "SeniorCitizen": [0, 1, 0, 1],
        "gender": ["Male", "Female", "Male", "Female"]
    })
    cleaned = clean_dataset(df)

    # " " and "" should be imputed with the median of [10.5, 20.0] which is 15.25
    assert cleaned["TotalCharges"].iloc[1] == 15.25
    assert cleaned["TotalCharges"].iloc[2] == 15.25
    assert cleaned["TotalCharges"].dtype == float

def test_clean_dataset_senior_citizen():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "SeniorCitizen": [0, 1, 0, 1]
    })
    cleaned = clean_dataset(df)
    assert cleaned["SeniorCitizen"].tolist() == ["No", "Yes", "No", "Yes"]

def test_clean_dataset_whitespace_strip():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "Contract": ["Month-to-month ", " One year", "Two year"]
    })
    cleaned = clean_dataset(df)
    assert cleaned["Contract"].tolist() == ["Month-to-month", "One year", "Two year"]

def test_clean_dataset_missing_categorical():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "gender": ["Male", np.nan, "Female", "Male"]
    })
    cleaned = clean_dataset(df)
    # Mode is Male
    assert cleaned["gender"].iloc[1] == "Male"

def test_encode_target():
    df = pd.DataFrame({
        "Churn": ["Yes", "No", "Yes"]
    })
    encoded = encode_target(df)
    assert encoded["Churn"].tolist() == [1, 0, 1]

def test_encode_target_invalid():
    df = pd.DataFrame({
        "Churn": ["Yes", "Maybe", "No"]
    })
    with pytest.raises(ValueError):
        encode_target(df)
