from src.utils.schema import SystemOutput
from src.model.inference import predict
from src.decision.engine import run_decision_engine
from src.explanation.generator import generate_explanation


def run_full_pipeline(context) -> SystemOutput:
    # step1 risk + uncertainty
    out = predict(context)
    risk = out["risk_probability"]
    unc = out["uncertainty_score"]
    conf = out["confidence_level"]

    # step2 decision
    dec = run_decision_engine(context, risk, conf)

    # setp3 explanation
    expl = generate_explanation(
        context=context,
        risk_probability=risk,
        confidence_level=conf,
        uncertainty_score=unc,
        decision_category=dec["decision_category"],
        reasoning_tags=dec["reasoning_tags"],
        shap_values=None,
    )
    return SystemOutput(
        patient_id=context.patient_id,
        window_index=context.window_index,
        risk_probability=risk,
        uncertainty_score=unc,
        confidence_level=conf,
        decision_category=dec["decision_category"],
        urgency=dec["urgency"],
        recommended_tests=dec["recommended_tests"],
        explanation=expl,
        context_summary={
            "physiological_state": context.physiological_state,
            "dominant_trends": [
                f
                for f, t in context.trends.items()
                if t in ("rising_fast", "falling_fast", "rising", "falling")
            ][:5],
            "data_confidence": context.data_confidence,
            "missing_critical": context.missing_critical,
        },
    )
