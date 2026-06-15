"""
app.py — Fibroid Risk Prediction Web App
=========================================
Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from data_loader import (
    parse_blood_pressure,
    encode_symptoms,
)
from train_and_save import train

app = Flask(__name__)

MODEL_PATH = "fibroid_model.pkl"
DATA_PATH = "augmented_fibroid_data.csv"

# Auto-train on first run if the model file isn't there yet
if not os.path.exists(MODEL_PATH):
    print("No saved model found — training a new one (first run only)...")
    train(DATA_PATH, MODEL_PATH)

model = joblib.load(MODEL_PATH)

# Exact feature order the model was trained on
FEATURE_COLUMNS = [
    "age", "height", "weight", "systolic_bp", "diastolic_bp",
    "symptom_bleeding", "symptom_heavy_period", "symptom_lower_abdominal_pain",
    "symptom_abdominal_pain", "symptom_painful_period", "symptom_frequent_urination",
    "symptom_constipation", "symptom_pain_during_sex", "symptom_body_weakness",
    "has_any_symptom", "bmi",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        age          = float(data["age"])
        height_m     = float(data["height"])
        weight_kg    = float(data["weight"])
        systolic_bp  = float(data["systolic_bp"])
        diastolic_bp = float(data["diastolic_bp"])
        symptoms     = set(data.get("symptoms", []))
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    bmi = weight_kg / (height_m ** 2)

    row = {
        "age": age, "height": height_m, "weight": weight_kg,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
        "symptom_bleeding":           int("symptom_bleeding" in symptoms),
        "symptom_heavy_period":       int("symptom_heavy_period" in symptoms),
        "symptom_lower_abdominal_pain": int("symptom_lower_abdominal_pain" in symptoms),
        "symptom_abdominal_pain":     int("symptom_abdominal_pain" in symptoms),
        "symptom_painful_period":     int("symptom_painful_period" in symptoms),
        "symptom_frequent_urination": int("symptom_frequent_urination" in symptoms),
        "symptom_constipation":       int("symptom_constipation" in symptoms),
        "symptom_pain_during_sex":    int("symptom_pain_during_sex" in symptoms),
        "symptom_body_weakness":      int("symptom_body_weakness" in symptoms),
        "has_any_symptom":            int(len(symptoms) > 0),
        "bmi": bmi,
    }

    X_input = pd.DataFrame([row])[FEATURE_COLUMNS]

    prediction  = int(model.predict(X_input)[0])
    probability = float(model.predict_proba(X_input)[0][prediction])

    risk_label = "HAS FIBROID RISK" if prediction == 1 else "LOW FIBROID RISK"
    risk_level = (
        "High" if prediction == 1 and probability >= 0.75 else
        "Moderate" if prediction == 1 else
        "Low"
    )

    return jsonify({
        "prediction":  prediction,
        "risk_label":  risk_label,
        "risk_level":  risk_level,
        "probability": round(probability * 100, 1),
        "bmi":         round(bmi, 1),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
