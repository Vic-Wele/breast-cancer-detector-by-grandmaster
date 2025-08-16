import sys
import pandas as pd
import joblib

def main(csv_path):
    # Load the model
    pipe = joblib.load("models/model.joblib")

    # Load the input data
    df = pd.read_csv(csv_path)

    # Ensure all required features exist
    missing = [c for c in pipe.feature_names_in_ if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in input data: {missing}")

    # Make predictions
    pred_class = pipe.predict(df[pipe.feature_names_in_])
    prob_benign = pipe.predict_proba(df[pipe.feature_names_in_])[:, 1]

    out = df.copy()
    out['pred_class'] = pred_class
    out['prob_benign'] = prob_benign
    out.to_csv("predictions.csv", index=False)
   
    # Print the predictions
    print("Predictions complete. Saved to predictions.csv")
    print(out.head())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])