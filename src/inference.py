"""
============================================================================
INFERENCE SCRIPT - inference.py
============================================================================
"""

import joblib
import os
import json
import numpy as np

# -------------------------------------------------------------------------
# FEATURE ORDER — MUST MATCH TRAINING EXACTLY
# -------------------------------------------------------------------------
FEATURE_ORDER = [
    "gender_male",
    "age",
    "monthly_charges",
    "total_charges",
    "tenure",
    "contract_month_to_month",
    "contract_two_year",
    "internet_dsl",
    "tech_support_yes",
    "streaming_tv_yes",
    "payment_electronic_check"
]

# -------------------------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------------------------
def model_fn(model_dir):
    print(f"Loading model from {model_dir}")
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print("✓ Model loaded")
    return model

# -------------------------------------------------------------------------
# INPUT PARSING
# -------------------------------------------------------------------------
def input_fn(request_body, content_type="application/json"):
    print(f"Content-Type: {content_type}")

    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")

    try:
        features = np.array([[data[f] for f in FEATURE_ORDER]], dtype=float)
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}")

    print(f"✓ Parsed input with {features.shape[1]} features")
    return features

# -------------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------------
def predict_fn(input_data, model):
    prob = model.predict_proba(input_data)[0, 1]
    return prob

# -------------------------------------------------------------------------
# OUTPUT FORMAT
# -------------------------------------------------------------------------
def output_fn(prediction, accept="application/json"):
    response = {
        "churn_probability": float(prediction),
        "prediction": "churn" if prediction >= 0.5 else "no_churn",
        "risk_level": (
            "high" if prediction >= 0.7
            else "medium" if prediction >= 0.4
            else "low"
        )
    }

    return json.dumps(response)
