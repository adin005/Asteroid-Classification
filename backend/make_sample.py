from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = r"C:\Users\nidas\Documents\Asteroid_Dataset\dataset.csv"
F = ["e", "a", "q", "i", "tp", "H", "diameter", "albedo", "moid"]

df = pd.read_csv(DATASET, usecols=F + ["pha"], low_memory=False)
for c in F:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# first hazardous asteroid that has all 9 values filled in
row = df[df["pha"].astype(str).str.upper() == "Y"].dropna(subset=F).head(1)
payload = {k: float(v) for k, v in row[F].iloc[0].items()}

row[F].to_csv(ROOT / "data" / "sample_inputs.csv", index=False)
(ROOT / "backend" / "test_payload.json").write_text(json.dumps(payload, indent=2))
print("Sample asteroid:", payload)

bundle = joblib.load(ROOT / "backend" / "models" / "model_bundle.joblib")
frame = pd.DataFrame([payload], columns=bundle["features"])
print("True label: hazardous (Y)")
print("Model prediction:", bundle["pipeline"].predict(frame).tolist(), "(1 = hazardous)")
print("Probabilities [non-hazardous, hazardous]:", bundle["pipeline"].predict_proba(frame).tolist())