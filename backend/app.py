from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models" / "model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
OPTIONAL = {"diameter", "albedo"}   # often missing in real data; the model fills them in

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch", missing=missing, extra=extra), 400

    row = []
    for f in FEATURES:
        value = data[f]
        if value is None or value == "":
            if f in OPTIONAL:
                row.append(np.nan)
                continue
            return jsonify(error=f"Missing value for {f}"), 400
        try:
            row.append(float(value))
        except (TypeError, ValueError):
            return jsonify(error=f"{f} must be a number"), 400

    try:
        X = np.array([row], dtype=float)
        prediction = int(MODEL.predict(X)[0])
        proba = MODEL.predict_proba(X)[0].tolist()
        return jsonify(
            prediction=prediction,
            label="Potentially hazardous" if prediction == 1 else "Not hazardous",
            probabilities=proba,
            hazardous_probability=proba[1],
            disclaimer="Research/teaching prototype only.",
        )
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)