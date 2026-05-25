import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="BRFSS Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_rf_pipeline.pkl"
METRICS_PATH = BASE_DIR / "models" / "final_model_metrics.csv"

# -----------------------------
# Load Model and Metrics
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)

model = load_model()
metrics = load_metrics()
# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Project Information")

    st.write(
        """
        This project predicts diabetes risk using selected health,
        lifestyle, and demographic features from the BRFSS 2024 dataset.
        """
    )

    st.markdown("**Final Model:** Tuned Random Forest")
    st.markdown("**Model Type:** Classification")
    st.markdown("**Decision Threshold:** 50%")
    st.markdown("**Number of Features:** 13")

    st.divider()

    st.subheader("Final Model Metrics")

    metrics_display = metrics.copy()
    metrics_display["Score"] = metrics_display["Score"].round(4)

    st.dataframe(metrics_display, use_container_width=True)

    st.divider()

    st.subheader("Important Note")
    st.caption(
        "This app is for educational purposes only and should not be used as a medical diagnosis tool."
    )
# -----------------------------
# Feature Labels
# -----------------------------
AGE_GROUP_LABELS = {
    1: "18 to 24",
    2: "25 to 29",
    3: "30 to 34",
    4: "35 to 39",
    5: "40 to 44",
    6: "45 to 49",
    7: "50 to 54",
    8: "55 to 59",
    9: "60 to 64",
    10: "65 to 69",
    11: "70 to 74",
    12: "75 to 79",
    13: "80 or older",
    14: "Unknown / Refused / Missing"
}

INCOME_LABELS = {
    1: "Less than $15,000",
    2: "$15,000 to < $25,000",
    3: "$25,000 to < $35,000",
    4: "$35,000 to < $50,000",
    5: "$50,000 to < $100,000",
    6: "$100,000 to < $200,000",
    7: "$200,000 or more"
}

# -----------------------------
# App UI
# -----------------------------
st.title("🩺 BRFSS Diabetes Prediction App")
st.warning(
    "This app is for educational purposes only and should not be used as a medical diagnosis tool."
)
st.write(
    """
    This app uses a trained machine learning pipeline to predict diabetes risk
    based on selected health, lifestyle, and demographic factors from the BRFSS 2024 dataset.
    """
)

st.success("Model loaded successfully! The final pipeline is ready for prediction.")
with st.expander("Project Summary"):
    st.write(
        """
        This project uses the BRFSS 2024 health survey dataset to build a diabetes prediction model.
        
        The workflow includes data loading, cleaning, exploratory data analysis, preprocessing,
        model training, model evaluation, hyperparameter tuning, and deployment using Streamlit.
        
        The final selected model is a tuned Random Forest pipeline that includes both preprocessing
        and classification steps.
        """
    )
# -----------------------------
# Prediction Form
# -----------------------------
st.divider()

st.subheader("Diabetes Risk Prediction")

st.write(
    """
    Enter the respondent health information below, then click **Predict**.
    """
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=28.0,
            step=0.1
        )

        general_health = st.selectbox(
            "General Health",
            options=[1, 0],
            format_func=lambda x: "Good / Better Health" if x == 1 else "Fair / Poor Health"
        )

        physical_health_days = st.slider(
            "Physical Health Days",
            min_value=0,
            max_value=30,
            value=0
        )

        mental_health_days = st.slider(
            "Mental Health Days",
            min_value=0,
            max_value=30,
            value=0
        )

        stroke = st.selectbox(
            "Stroke",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:
        heart_disease = st.selectbox(
            "Heart Disease",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        smoker = st.selectbox(
            "Smoking Status",
            options=[0, 1],
            format_func=lambda x: "Never Smoker" if x == 0 else "Smoker / Former Smoker"
        )

        physical_activity = st.selectbox(
            "Physical Activity",
            options=[1, 0],
            format_func=lambda x: "Had Physical Activity" if x == 1 else "No Physical Activity"
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

    with col3:
        age = st.selectbox(
    "Age Group",
    options=list(AGE_GROUP_LABELS.keys()),
    index=7,
    format_func=lambda x: AGE_GROUP_LABELS[x]
)

        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Did not graduate High School",
                2: "Graduated High School",
                3: "Attended College / Technical School",
                4: "Graduated College / Technical School"
            }[x]
        )

        income = st.selectbox(
    "Income Level",
    options=list(INCOME_LABELS.keys()),
    index=4,
    format_func=lambda x: INCOME_LABELS[x]
)

        heavy_alcohol = st.selectbox(
            "Heavy Alcohol Consumption",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    submitted = st.form_submit_button("Predict Diabetes Risk")
with st.expander("Feature Guide"):
    st.write("**Age Group:** Based on BRFSS age categories.")
    st.write("**Income Level:** Based on BRFSS income categories.")
    st.write("**Physical / Mental Health Days:** Number of unhealthy days during the past 30 days.")
    st.write("**Prediction Probability:** The model's estimated probability for the diabetes class.")
    
if submitted:
    input_data = pd.DataFrame([{
        "BMI": bmi,
        "GeneralHealth": general_health,
        "PhysicalHealthDays": physical_health_days,
        "MentalHealthDays": mental_health_days,
        "Stroke": stroke,
        "HeartDisease": heart_disease,
        "Smoker": smoker,
        "PhysicalActivity": physical_activity,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income,
        "HeavyAlcohol": heavy_alcohol
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Higher Diabetes Risk")
    else:
        st.success("Prediction: Lower Diabetes Risk")

    probability_percentage = probability * 100

    st.metric("Diabetes Probability", f"{probability_percentage:.2f}%")

    st.progress(float(probability))

    if probability >= 0.5:
        st.write("The probability is above the 50% decision threshold.")
    else:
        st.write("The probability is below the 50% decision threshold.")

    st.caption(
    "Note: This prediction is based on a machine learning model and should be used for educational purposes only, not medical diagnosis."
)

    with st.expander("Show input data"):
        st.dataframe(input_data, use_container_width=True)
        
# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "BRFSS Diabetes Prediction Project | Built with Python, Scikit-learn, and Streamlit"
)