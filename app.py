"""
app.py
------
Flask web service that serves salary predictions via:
  GET  /           → HTML frontend
  POST /predict    → JSON API  { experience, education, role, skills_count }
                             → { predicted_salary, formatted, confidence }

Author  : C. Thrishitha
Project : Employee Salary Prediction Web Service
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# ── Load trained artifacts ────────────────────────────────────────────────────
model  = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

# ── Lookup tables (must match train_model.py encoding) ───────────────────────
EDUCATION_MAP = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
ROLE_MAP      = {0: "Junior",      1: "Mid-Level",  2: "Senior",   3: "Lead/Manager"}


@app.route("/")
def home():
    """Render the HTML prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body:
        { "experience": int, "education": int, "role": int, "skills_count": int }
    Returns JSON:
        { "predicted_salary": float, "formatted": str, "confidence": str }
    """
    try:
        data = request.get_json(force=True)

        experience    = int(data["experience"])
        education     = int(data["education"])
        role          = int(data["role"])
        skills_count  = int(data["skills_count"])

        # Validate ranges
        assert 0 <= experience   <= 40,  "Experience must be 0–40 years."
        assert education  in EDUCATION_MAP, "Invalid education code (0-3)."
        assert role       in ROLE_MAP,      "Invalid role code (0-3)."
        assert 1 <= skills_count <= 20,  "Skills count must be 1–20."

        features = np.array([[experience, education, role, skills_count]])
        features_scaled = scaler.transform(features)

        salary = model.predict(features_scaled)[0]

        # Derive all tree predictions for a confidence range
        tree_preds = np.array([tree.predict(features_scaled)[0]
                                for tree in model.estimators_])
        low  = np.percentile(tree_preds, 10)
        high = np.percentile(tree_preds, 90)

        return jsonify({
            "predicted_salary": round(float(salary), 2),
            "formatted":        f"₹{salary:,.0f} per year",
            "confidence_range": f"₹{low:,.0f} – ₹{high:,.0f}",
            "education_label":  EDUCATION_MAP[education],
            "role_label":       ROLE_MAP[role],
        })

    except (KeyError, ValueError, AssertionError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error: " + str(e)}), 500


if __name__ == "__main__":
    # Development server — use gunicorn for production
    app.run(host="0.0.0.0", port=5000, debug=True)
