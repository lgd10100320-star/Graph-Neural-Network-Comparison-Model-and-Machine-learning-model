from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "catboost_model.pkl"
SCALER_PATH = BASE_DIR / "catboost_scaler.pkl"
INPUT_PATH = BASE_DIR / "last.csv"
OUTPUT_PATH = BASE_DIR / "catboost_predictions_results_last.csv"

catboost_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

new_data = pd.read_csv(INPUT_PATH)

if new_data.shape[1] < 25:
    raise ValueError("Input data must contain at least 25 feature columns.")

if new_data.shape[1] == 25:
    x_new = new_data.iloc[:, :]
else:
    x_new = new_data.iloc[:, 1:26]

x_new_scaled = scaler.transform(x_new)
predictions = catboost_model.predict(x_new_scaled)

new_data["Predicted_PCE"] = predictions
new_data.to_csv(OUTPUT_PATH, index=False)

print(f"CatBoost prediction completed. Results saved to {OUTPUT_PATH.name}")
print("\nCatBoost prediction preview:")
print(new_data[["Predicted_PCE"]].head())

print("\nPrediction summary:")
print(f"Sample count: {len(predictions)}")
print(f"PCE range: [{predictions.min():.3f}, {predictions.max():.3f}]")
print(f"Mean PCE: {predictions.mean():.3f}")
print(f"Standard deviation: {predictions.std():.3f}")
