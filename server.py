from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
INTERFACE_DIR = BASE_DIR / "interface"
PIPELINE_PATH = BASE_DIR / "artifacts" / "attrition_pipeline.pkl"

app = Flask(__name__)

pipeline = joblib.load(PIPELINE_PATH)

INPUT_FIELDS = [
    "client_gender",
    "senior_flag",
    "has_partner",
    "has_dependents",
    "tenure_months",
    "phone_line",
    "multi_line",
    "internet_plan",
    "security_addon",
    "backup_addon",
    "protection_addon",
    "support_addon",
    "tv_streaming",
    "movie_streaming",
    "contract_term",
    "ebilling",
    "payment_mode",
    "monthly_fee",
    "total_billed",
    "mean_monthly_fee",
    "active_service_count",
]


@app.route("/")
def index():
    wants_html = (
        request.accept_mimetypes.best_match(["application/json", "text/html"])
        == "text/html"
    )

    if wants_html:
        return send_from_directory(INTERFACE_DIR, "console.html")

    return jsonify({
        "service": "Telco Attrition Predictor",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def assess_customer():

    try:

        payload = request.get_json(silent=True)

        if not payload or not isinstance(payload, dict):
            return jsonify({
                "error": "Request body must contain JSON data"
            }), 400

        missing_fields = [
            field
            for field in INPUT_FIELDS
            if field not in payload
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        record = pd.DataFrame([payload])

        label = pipeline.predict(record)[0]

        probability = pipeline.predict_proba(record)[0][1]

        return jsonify({
            "attrition_flag": "Yes" if label == 1 else "No",
            "attrition_probability": round(float(probability), 4)
        })

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
