TREND_PHRASES = {
    "rising_fast": "rapidly rising",
    "rising": "rising steadily",
    "stable": "stable",
    "falling": "falling steadily",
    "falling_fast": "falling rapidly",
    "unknown": "unclear (insufficient data)",
}
FEATURE_NAMES = {
    "heart_rate": "heart rate",
    "systolic_bp": "systolic blood pressure",
    "diastolic_bp": "diastolic blood pressure",
    "respiratory_rate": "respiratory rate",
    "temperature_c": "temperature",
    "spo2": "oxygen saturation",
    "lactate": "lactate",
    "wbc": "white blood cell count",
    "creatinine": "creatinine",
    "glucose": "blood glucose",
    "hemoglobin": "hemoglobin",
}
STATE_PHRASES = {
    "stable": "The overall physiological trajectory is stable.",
    "concerning": "The overall trajectory shows mild concern.",
    "deteriorating": "The overall trajectory indicates deterioration.",
    "critical": "The overall trajectory is critically concerning.",
}
ACTION_PHRASES = {
    "ESCALATE_IMMEDIATELY": "Immediate clinical escalation is recommended.",
    "ORDER_ADDITIONAL_TESTS": "Additional diagnostic tests are recommended before escalation.",
    "MONITOR_CLOSELY": "Close monitoring with reassessment is recommended.",
    "CONTINUE_MONITORING": "Continued standard monitoring is appropriate at this time.",
    "GATHER_MORE_DATA": "Gathering additional measurements is recommended to reduce uncertainty.",
    "MONITOR_AND_PREPARE": "Active monitoring with preparation for escalation is advised.",
}


def generate_explanation(
    context,
    risk_probability,
    confidence_level,
    uncertainty_score,
    decision_category,
    reasoning_tags,
    shap_values=None,
) -> str:
    sentences = []

    # s1 risk drivers
    drivers = _get_drivers(context, shap_values)
    if drivers:
        sentences.append(
            f'The elevated risk prediction is driven by {", ".join(drivers[:3])}.'
        )

    # s2 trends
    concerning = [
        (f, TREND_PHRASES[t])
        for f, t in context.trends.items()
        if t in ("rising", "rising_fast", "falling", "falling_fast")
        and f in FEATURE_NAMES
    ]
    if concerning[:2]:
        parts = [f"{FEATURE_NAMES[f]} is {t}" for f, t in concerning[:2]]
        sentences.append("Over the observation window, " + " and ".join(parts) + ".")

    # s3 state
    sentences.append(STATE_PHRASES.get(context.physiological_state, ""))

    # s4: uncertainty
    if confidence_level in ("Medium", "Low"):
        if context.missing_critical:
            mf = " and ".join(FEATURE_NAMES.get(f, f) for f in context.missing_critical)
            sentences.append(
                f"Confidence is {confidence_level.lower()} because {mf} data is missing, limiting trend certainty."
            )
        else:
            sentences.append(
                f"Confidence is {confidence_level.lower()} due to sparse data (density: {context.observation_density:.0%})."
            )

    # s5: action
    sentences.append(
        ACTION_PHRASES.get(decision_category, "Clinical judgment is advised.")
    )
    return " ".join(s for s in sentences if s)


def _get_drivers(context, shap_values):
    if shap_values:
        top = [k for k, v in list(shap_values.items())[:5] if v > 0]
        return [
            f'{FEATURE_NAMES.get(f.split("_")[0], f)} ({TREND_PHRASES.get(context.trends.get(f.split("_")[0], "unknown"), "")})'
            for f in top[:3]
        ]
    return [
        f"{FEATURE_NAMES.get(f, f)} ({TREND_PHRASES.get(t, t)})"
        for f, t in context.trends.items()
        if t in ("rising_fast", "rising") and f in FEATURE_NAMES
    ][:3]
