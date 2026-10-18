import argparse
from pathlib import Path
import pandas as pd
from cricket_predictor.modeling import train_models

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Train cricket match outcome prediction models.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample_matches.csv",
                        help="Historical match CSV (default: bundled synthetic sample).")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts",
                        help="Directory for the saved model, scores and charts.")
    parser.add_argument("--window", type=int, default=5, help="Number of earlier matches per team's recent form.")
    args = parser.parse_args()
    try:
        artifact = train_models(args.data, args.output, window=args.window)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    metrics = artifact["metrics"]
    if artifact["metadata"]["synthetic_data"]:
        print("DEMO: synthetic data. These scores do not establish real-world prediction accuracy.")
    print(pd.DataFrame(metrics["candidates"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSelected on validation: {metrics['selected_model']}")
    print(f"Held-out test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Majority-class baseline: {metrics['baseline_test_accuracy']:.2%}")
    print(f"Cleaned matches: {metrics['cleaning_report']['output_rows']}")
    print(f"Saved all-history refit to: {(args.output / 'best_model.joblib').resolve()}")
    print("Start the UI: python -m streamlit run app.py")

if __name__ == "__main__":
    main()