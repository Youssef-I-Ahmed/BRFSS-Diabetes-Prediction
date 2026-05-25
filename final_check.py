import os
import joblib
import pandas as pd

print("=" * 50)
print("FINAL PROJECT CHECK")
print("=" * 50)

# -----------------------------
# 1. Check required files
# -----------------------------
required_files = [
    "app.py",
    "README.md",
    "requirements.txt",
    "data/brfss_diabetes_cleaned.csv",
    "models/diabetes_rf_pipeline.pkl",
    "models/final_model_metrics.csv",
    "notebooks/01_brfss_diabetes_analysis.ipynb",
    "notebooks/02_eda_analysis.ipynb",
    "notebooks/03_modeling_pipeline.ipynb",
    "notebooks/04_model_evaluation_saving.ipynb",
]

print("\n1. Checking required files:")

all_files_exist = True

for file in required_files:
    exists = os.path.exists(file)
    print(f"{file} => {exists}")

    if not exists:
        all_files_exist = False

# -----------------------------
# 2. Check imports
# -----------------------------
print("\n2. Checking imports:")

try:
    import numpy
    import sklearn
    import streamlit
    import matplotlib
    print("All required packages imported successfully!")
    imports_ok = True
except Exception as e:
    print("Import error:", e)
    imports_ok = False

# -----------------------------
# 3. Load model and metrics
# -----------------------------
print("\n3. Loading model and metrics:")

try:
    model = joblib.load("models/diabetes_rf_pipeline.pkl")
    metrics = pd.read_csv("models/final_model_metrics.csv")

    print("Model loaded successfully!")
    print("Metrics loaded successfully!")
    print("Model type:", type(model))
    print("\nMetrics:")
    print(metrics)

    model_ok = True
except Exception as e:
    print("Model or metrics loading error:", e)
    model_ok = False

# -----------------------------
# 4. Test prediction
# -----------------------------
print("\n4. Testing prediction:")

try:
    sample_input = pd.DataFrame([{
        "BMI": 28.0,
        "GeneralHealth": 1,
        "PhysicalHealthDays": 0,
        "MentalHealthDays": 0,
        "Stroke": 0,
        "HeartDisease": 0,
        "Smoker": 0,
        "PhysicalActivity": 1,
        "Sex": 0,
        "Age": 8,
        "Education": 4,
        "Income": 5,
        "HeavyAlcohol": 0
    }])

    prediction = model.predict(sample_input)[0]
    probability = model.predict_proba(sample_input)[0][1]

    print("Prediction completed successfully!")
    print("Prediction:", prediction)
    print("Diabetes probability:", round(probability, 4))

    prediction_ok = True
except Exception as e:
    print("Prediction error:", e)
    prediction_ok = False

# -----------------------------
# 5. Final result
# -----------------------------
print("\n" + "=" * 50)
print("FINAL CHECK RESULT")
print("=" * 50)

if all_files_exist and imports_ok and model_ok and prediction_ok:
    print("✅ Project is ready!")
else:
    print("❌ Some checks failed. Review the messages above.")