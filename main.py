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


if __name__ == "__main__":
    main()
