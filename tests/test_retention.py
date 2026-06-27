from src.retention_recommendations import generate_recommendations, get_risk_level

def test_get_risk_level():
    assert get_risk_level(0.1) == "Low"
    assert get_risk_level(0.35) == "Medium"
    assert get_risk_level(0.5) == "Medium"
    assert get_risk_level(0.65) == "High"
    assert get_risk_level(0.9) == "High"

def test_generate_recommendations_high_risk_mtm():
    # 1. High churn probability + month-to-month contract.
    customer = {
        "Contract": "Month-to-month",
        "TechSupport": "Yes",
        "MonthlyCharges": 50,
        "tenure": 24
    }
    recs = generate_recommendations(0.8, customer)
    categories = [r["category"] for r in recs]
    assert "📋 Contract Incentive" in categories
    assert recs[0]["priority"] == 1

def test_generate_recommendations_high_risk_no_tech():
    # 2. High churn probability + no TechSupport.
    customer = {
        "Contract": "One year",
        "TechSupport": "No",
        "MonthlyCharges": 50,
        "tenure": 24
    }
    recs = generate_recommendations(0.8, customer)
    categories = [r["category"] for r in recs]
    assert "🛠️ Technical Support" in categories

def test_generate_recommendations_high_risk_high_charge():
    # 3. High churn probability + high monthly charge.
    customer = {
        "Contract": "One year",
        "TechSupport": "Yes",
        "MonthlyCharges": 100.0,
        "tenure": 24
    }
    recs = generate_recommendations(0.8, customer)
    categories = [r["category"] for r in recs]
    assert "💰 Plan Review" in categories

def test_generate_recommendations_high_risk_short_tenure():
    # 4. High churn probability + short tenure.
    customer = {
        "Contract": "One year",
        "TechSupport": "Yes",
        "MonthlyCharges": 50.0,
        "tenure": 6
    }
    recs = generate_recommendations(0.8, customer)
    categories = [r["category"] for r in recs]
    assert "🎯 Onboarding Support" in categories

def test_generate_recommendations_medium_risk():
    # 5. Medium-risk customer.
    customer = {
        "Contract": "One year",
        "TechSupport": "Yes",
        "MonthlyCharges": 50.0,
        "tenure": 24
    }
    recs = generate_recommendations(0.5, customer)
    categories = [r["category"] for r in recs]
    assert "📞 Proactive Check-in" in categories

def test_generate_recommendations_low_risk_long_tenure():
    # 6. Low-risk long-tenure customer with a long-term contract.
    customer = {
        "Contract": "Two year",
        "TechSupport": "Yes",
        "MonthlyCharges": 50.0,
        "tenure": 60
    }
    recs = generate_recommendations(0.1, customer)
    categories = [r["category"] for r in recs]
    assert "🌟 Loyalty Engagement" in categories
    assert "🏆 Long-term Loyalty" in categories
