from src.decision.rules import classify_risk_level, DECISION_MATRIX, get_default

WORSENING = {"deteriorating", "critical"}


def run_decision_engine(context, risk_probability, confidence_level) -> dict:
    risk_level = classify_risk_level(risk_probability)
    trend_state = "worsening" if context.physiological_state in WORSENING else "stable"
    key = (risk_level, confidence_level, trend_state)
    decision = DECISION_MATRIX.get(key) or DECISION_MATRIX.get(
        (risk_level, confidence_level, "stable"), get_default()
    )
    return {
        "decision_category": decision["category"],
        "urgency": decision["urgency"],
        "recommended_tests": decision.get("tests", []),
        "reassessment_window": decision.get("reassess", "60_minutes"),
        "reasoning_tags": _build_tags(context, risk_level, confidence_level),
    }


def _build_tags(context, risk_level, confidence_level):
    tags = [f"risk_{risk_level}", f"confidence_{confidence_level.lower()}"]
    if context.deterioration_flag:
        tags.append("multi_feature_deterioration")
    tags += [f"missing_{f}" for f in context.missing_critical]
    if context.physiological_state in WORSENING:
        tags.append(f"state_{context.physiological_state}")
    for feat, trend in context.trends.items():
        if trend in ("rising_fast", "falling_fast"):
            tags.append(f"{feat}_{trend}")
    return tags
