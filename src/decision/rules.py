from src.utils.config_loader import get_config

cfg = get_config()
RISK_HIGH = cfg["decision"]["risk_high_threshold"]
RISK_MOD = cfg["decision"]["risk_moderate_threshold"]


def classify_risk_level(prob):
    if prob >= RISK_HIGH:
        return "high"
    if prob >= RISK_MOD:
        return "moderate"
    return "low"


# Key: (risk_level, confidence_level, trend_state)
# confidence_level is 'High'/'Medium'/'Low' from model
# trend_state is 'worsening' or 'stable'
DECISION_MATRIX = {
    ("high", "High", "worsening"): {
        "category": "ESCALATE_IMMEDIATELY",
        "urgency": "immediate",
        "tests": [],
        "reassess": "15_minutes",
    },
    ("high", "High", "stable"): {
        "category": "MONITOR_AND_PREPARE",
        "urgency": "urgent",
        "tests": [],
        "reassess": "30_minutes",
    },
    ("high", "Medium", "worsening"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "urgent",
        "tests": ["lactate_repeat", "blood_culture", "abg"],
        "reassess": "30_minutes",
    },
    ("high", "Medium", "stable"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "urgent",
        "tests": ["lactate_repeat", "cbc"],
        "reassess": "30_minutes",
    },
    ("high", "Low", "worsening"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "urgent",
        "tests": ["lactate_repeat", "blood_culture", "cxr"],
        "reassess": "20_minutes",
    },
    ("high", "Low", "stable"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "urgent",
        "tests": ["lactate_repeat"],
        "reassess": "30_minutes",
    },
    ("moderate", "High", "worsening"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "routine",
        "tests": ["lactate", "cbc"],
        "reassess": "30_minutes",
    },
    ("moderate", "High", "stable"): {
        "category": "MONITOR_CLOSELY",
        "urgency": "routine",
        "tests": [],
        "reassess": "60_minutes",
    },
    ("moderate", "Medium", "worsening"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "routine",
        "tests": ["lactate"],
        "reassess": "30_minutes",
    },
    ("moderate", "Medium", "stable"): {
        "category": "MONITOR_CLOSELY",
        "urgency": "routine",
        "tests": [],
        "reassess": "60_minutes",
    },
    ("moderate", "Low", "worsening"): {
        "category": "ORDER_ADDITIONAL_TESTS",
        "urgency": "routine",
        "tests": ["lactate"],
        "reassess": "45_minutes",
    },
    ("low", "High", "stable"): {
        "category": "CONTINUE_MONITORING",
        "urgency": "routine",
        "tests": [],
        "reassess": "60_minutes",
    },
    ("low", "Low", "stable"): {
        "category": "GATHER_MORE_DATA",
        "urgency": "routine",
        "tests": ["vitals_reassessment"],
        "reassess": "60_minutes",
    },
}


def get_default():
    return {
        "category": "MONITOR_CLOSELY",
        "urgency": "routine",
        "tests": [],
        "reassess": "60_minutes",
    }
