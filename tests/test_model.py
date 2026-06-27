import pytest
from src.utils import load_pickle
from src.config import MODEL_PATH
import pandas as pd
import numpy as np

@pytest.fixture
def model():
    # Only test if model exists
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found.")
    return load_pickle(MODEL_PATH)

def test_model_predict_proba(model):
    # Create a synthetic dataframe with expected features
    from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = [10.0]
    for col in CATEGORICAL_FEATURES:
        data[col] = ["No"]

    df = pd.DataFrame(data)

    proba = model.predict_proba(df)

    assert proba.shape == (1, 2)
    assert 0 <= proba[0, 1] <= 1
    assert np.isclose(proba[0, 0] + proba[0, 1], 1.0)

def test_model_predict(model):
    from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = [10.0]
    for col in CATEGORICAL_FEATURES:
        data[col] = ["No"]

    df = pd.DataFrame(data)

    pred = model.predict(df)
    assert pred[0] in [0, 1]
