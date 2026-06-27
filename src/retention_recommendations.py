"""
retention_recommendations.py — Rule-based retention recommendation engine.

Generates actionable, prioritised retention recommendations based on
the churn probability and key customer attributes.

These recommendations are decision-support guidance — they should not
automatically deny service, target customers unfairly, or replace
human review.
"""

from typing import Dict, List


def get_risk_level(probability: float) -> str:
    """
    Classify churn probability into risk levels.

    Parameters
    ----------
    probability : float
        Churn probability (0–1).

    Returns
    -------
    str
        'Low', 'Medium', or 'High'.
    """
    if probability >= 0.65:
        return "High"
    elif probability >= 0.35:
        return "Medium"
    else:
        return "Low"


def generate_recommendations(
    probability: float,
    customer_data: Dict,
) -> List[Dict]:
    """
    Generate prioritised retention recommendations.

    Parameters
    ----------
    probability : float
        Predicted churn probability (0–1).
    customer_data : dict
        Customer attributes (raw input values). Expected keys include:
        Contract, TechSupport, MonthlyCharges, tenure, OnlineSecurity,
        InternetService, Partner, Dependents.

    Returns
    -------
    list of dict
        Each dict has keys: priority (1=highest), category, recommendation, rationale.
    """
    risk_level = get_risk_level(probability)
    recommendations = []

    contract = customer_data.get("Contract", "")
    tech_support = customer_data.get("TechSupport", "")
    monthly_charges = customer_data.get("MonthlyCharges", 0)
    tenure = customer_data.get("tenure", 0)
    online_security = customer_data.get("OnlineSecurity", "")
    internet_service = customer_data.get("InternetService", "")
    partner = customer_data.get("Partner", "")
    dependents = customer_data.get("Dependents", "")

    if risk_level == "High":
        # ----- High risk: month-to-month contract -----
        if contract == "Month-to-month":
            recommendations.append({
                "priority": 1,
                "category": "📋 Contract Incentive",
                "recommendation": (
                    "Offer a personalised fixed-term contract incentive "
                    "(e.g., 15–20% discount for a 12-month commitment)."
                ),
                "rationale": (
                    "Month-to-month contracts have the lowest switching cost. "
                    "A competitive long-term offer may reduce churn intent."
                ),
            })

        # ----- High risk: no tech support -----
        if tech_support in ("No", "No internet service"):
            recommendations.append({
                "priority": 1,
                "category": "🛠️ Technical Support",
                "recommendation": (
                    "Offer a complimentary 3-month tech-support trial or "
                    "prioritise a service-quality outreach call."
                ),
                "rationale": (
                    "Customers without tech support may have unresolved issues "
                    "contributing to dissatisfaction."
                ),
            })

        # ----- High risk: no online security -----
        if online_security in ("No", "No internet service"):
            recommendations.append({
                "priority": 2,
                "category": "🔒 Security Services",
                "recommendation": (
                    "Recommend an online security add-on or bundle it into "
                    "a value package at a reduced rate."
                ),
                "rationale": (
                    "Adding security services can increase perceived value "
                    "and create additional switching barriers."
                ),
            })

        # ----- High risk: high monthly charges -----
        if monthly_charges > 70:
            recommendations.append({
                "priority": 1,
                "category": "💰 Plan Review",
                "recommendation": (
                    "Conduct a personalised plan review. Consider offering "
                    "a bundle discount, loyalty credit, or downgrade option "
                    "that better fits the customer's usage patterns."
                ),
                "rationale": (
                    "High monthly charges relative to perceived value is a "
                    "common churn trigger. A proactive price review signals "
                    "that the company values the customer."
                ),
            })

        # ----- High risk: low tenure (new customer) -----
        if tenure <= 12:
            recommendations.append({
                "priority": 1,
                "category": "🎯 Onboarding Support",
                "recommendation": (
                    "Activate the new-customer welcome programme: dedicated "
                    "account manager check-in, feature walkthrough, and a "
                    "first-year loyalty reward."
                ),
                "rationale": (
                    "New customers are in the critical adoption phase. "
                    "Early engagement and support significantly reduce "
                    "early-life churn."
                ),
            })

        # ----- High risk: fibre optic -----
        if internet_service == "Fiber optic":
            recommendations.append({
                "priority": 2,
                "category": "📡 Service Quality",
                "recommendation": (
                    "Schedule a proactive network quality assessment and "
                    "communicate any planned infrastructure improvements."
                ),
                "rationale": (
                    "Fibre optic customers tend to have higher expectations. "
                    "Proactive quality assurance can prevent dissatisfaction."
                ),
            })

        # ----- General high-risk fallback -----
        if not recommendations:
            recommendations.append({
                "priority": 1,
                "category": "⚠️ Retention Outreach",
                "recommendation": (
                    "Initiate a priority retention call with a trained "
                    "retention specialist. Prepare a personalised offer "
                    "based on the customer's usage history."
                ),
                "rationale": (
                    "This customer has a high predicted churn probability. "
                    "Direct human engagement is the most effective "
                    "retention intervention."
                ),
            })

    elif risk_level == "Medium":
        recommendations.append({
            "priority": 2,
            "category": "📞 Proactive Check-in",
            "recommendation": (
                "Schedule a customer-satisfaction check-in call or send "
                "a personalised satisfaction survey within the next 30 days."
            ),
            "rationale": (
                "Medium-risk customers may have emerging concerns. "
                "A proactive check-in can surface issues before they "
                "escalate to churn intent."
            ),
        })

        if contract == "Month-to-month":
            recommendations.append({
                "priority": 2,
                "category": "📋 Contract Upgrade",
                "recommendation": (
                    "Present a value-added annual contract option with "
                    "a moderate discount (10–15%)."
                ),
                "rationale": (
                    "Transitioning to a longer-term contract reduces "
                    "churn risk by increasing switching costs."
                ),
            })

        if monthly_charges > 70:
            recommendations.append({
                "priority": 3,
                "category": "💡 Value Communication",
                "recommendation": (
                    "Send a personalised usage report highlighting the "
                    "value received for the current plan."
                ),
                "rationale": (
                    "Reinforcing perceived value can reduce price sensitivity."
                ),
            })

    else:  # Low risk
        recommendations.append({
            "priority": 3,
            "category": "🌟 Loyalty Engagement",
            "recommendation": (
                "Maintain current service quality. Consider including "
                "the customer in a loyalty rewards programme or "
                "referral incentive scheme."
            ),
            "rationale": (
                "Low-risk customers are valuable. Positive engagement "
                "can further strengthen retention and generate referrals."
            ),
        })

        if tenure > 48:
            recommendations.append({
                "priority": 3,
                "category": "🏆 Long-term Loyalty",
                "recommendation": (
                    "Recognise the customer's long tenure with a loyalty "
                    "milestone reward or exclusive benefit."
                ),
                "rationale": (
                    "Acknowledging loyalty reinforces the relationship "
                    "and reduces the risk of competitor poaching."
                ),
            })

    # Sort by priority (1 = highest)
    recommendations.sort(key=lambda x: x["priority"])

    return recommendations


def format_risk_badge(probability: float) -> dict:
    """
    Return risk badge styling information.

    Parameters
    ----------
    probability : float
        Churn probability (0–1).

    Returns
    -------
    dict
        Keys: level, label, colour, icon.
    """
    risk = get_risk_level(probability)

    badges = {
        "High": {
            "level": "High",
            "label": "High Risk of Churn",
            "colour": "#E74C3C",
            "icon": "🔴",
        },
        "Medium": {
            "level": "Medium",
            "label": "Medium Risk of Churn",
            "colour": "#F39C12",
            "icon": "🟡",
        },
        "Low": {
            "level": "Low",
            "label": "Low Risk of Churn",
            "colour": "#27AE60",
            "icon": "🟢",
        },
    }

    return badges.get(risk, badges["Medium"])
