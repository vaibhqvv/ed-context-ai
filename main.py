import argparse
from src.utils.logger import get_logger

log = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="ED Decision Support")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["preprocess", "context", "train", "evaluate", "demo", "all"],
    )
    args = parser.parse_args()

    if args.phase in ("preprocess", "all"):
        from src.preprocessing.pipeline import run_preprocessing

        run_preprocessing()

    if args.phase in ("context", "all"):
        from src.context.builder import run_context_construction

        run_context_construction()

    if args.phase in ("train", "all"):
        from src.model.trainer import train

        train()

    if args.phase in ("evaluate", "all"):
        from src.evaluation.eval_runner import run_evaluation

        run_evaluation()

    if args.phase == "demo":
        import pickle
        from pathlib import Path
        from src.utils.config_loader import get_config
        from src.explanation.pipeline import run_full_pipeline

        cfg = get_config()
        with open(
            Path(cfg["paths"]["context_data"]) / "context_objects.pkl", "rb"
        ) as f:
            ctxs = pickle.load(f)
        result = run_full_pipeline(ctxs[0])
        print("\n=== SYSTEM OUTPUT ===")
        print(f"Patient:     {result.patient_id}")
        print(f"Risk:        {result.risk_probability:.3f}")
        print(
            f"Uncertainty: {result.uncertainty_score:.4f}  ({result.confidence_level})"
        )
        print(f"Decision:    {result.decision_category} [{result.urgency}]")
        print(f"Explanation:\n  {result.explanation}")


if __name__ == "__main__":
    main()
