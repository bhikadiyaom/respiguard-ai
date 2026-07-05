"""Train and save the RespiGuard-AI model.

Developer runs this once:  python train_model.py

Steps:
  1. Run the data cleaning pipeline (auto-detects real vs sample data).
  2. Run the feature engineering pipeline.
  3. Train and save the model (pgmpy primary, GaussianNB backup).
  4. Print a complete training summary.
"""

from src.data_cleaning import run_cleaning
from src.feature_engineering import run_feature_engineering
from src.bayesian_model import train_model


def main():
    """Orchestrate the full training workflow and print a summary."""
    # Step 1: clean and merge the datasets.
    print("\n[1/3] Running data cleaning pipeline...")
    merged_df, using_sample = run_cleaning()

    # Step 2: engineer features for the model.
    print("\n[2/3] Running feature engineering pipeline...")
    engineered_df = run_feature_engineering(merged_df)

    # Step 3: train and persist the model.
    print("\n[3/3] Training model...")
    _, accuracy, f1, model_type = train_model(engineered_df)

    # Step 4: final training summary.
    print("")
    print("Data mode: " + ("SAMPLE" if using_sample else "REAL"))
    print(f"Model type: {model_type}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"F1 score:   {f1:.3f}")
    print("========================================")
    print("RespiGuard-AI Model Training Complete")
    print("Model saved: models/trained_model_v1.pkl")
    print("Run the app: streamlit run app.py")
    print("========================================")


if __name__ == "__main__":
    main()
