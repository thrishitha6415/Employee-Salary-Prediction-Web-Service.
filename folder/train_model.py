"""
train_model.py
--------------
Trains a Random Forest Regressor to predict employee salaries.
Saves the trained model and feature scaler as .pkl files.

Author  : C. Thrishitha
Project : Employee Salary Prediction Web Service
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Generate realistic synthetic dataset ──────────────────────────────────
np.random.seed(42)
n = 500

experience     = np.random.randint(0, 20, n)                # years 0-19
education_map  = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
education      = np.random.choice([0, 1, 2, 3], n, p=[0.10, 0.55, 0.25, 0.10])
role_map       = {0: "Junior", 1: "Mid-Level", 2: "Senior", 3: "Lead/Manager"}
role           = np.random.choice([0, 1, 2, 3], n, p=[0.30, 0.35, 0.25, 0.10])
skills_count   = np.random.randint(1, 10, n)               # number of skills

# Salary formula (realistic)
salary = (
    25000
    + experience * 2800
    + education * 8000
    + role * 12000
    + skills_count * 500
    + np.random.normal(0, 3000, n)                         # noise
)
salary = np.clip(salary, 20000, 200000).round(-2)          # round to hundreds

df = pd.DataFrame({
    "experience":   experience,
    "education":    education,
    "role":         role,
    "skills_count": skills_count,
    "salary":       salary
})

# ── 2. Train / test split ─────────────────────────────────────────────────────
X = df[["experience", "education", "role", "skills_count"]]
y = df["salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
preds = model.predict(X_test_scaled)
print(f"MAE  : ₹{mean_absolute_error(y_test, preds):,.0f}")
print(f"R²   : {r2_score(y_test, preds):.4f}")

# ── 6. Save artifacts ────────────────────────────────────────────────────────
joblib.dump(model,  "salary_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✓ Model and scaler saved.")
