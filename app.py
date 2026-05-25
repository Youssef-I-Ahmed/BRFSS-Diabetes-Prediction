import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="BRFSS Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# Paths
# =========================================================
MODEL_PATH = "models/diabetes_rf_pipeline.pkl"
METRICS_PATH = "models/final_model_metrics.csv"
DATA_PATH = "data/brfss_diabetes_cleaned.csv"
LINKEDIN_URL = "https://www.linkedin.com/in/yousef-ismail87/"

# =========================================================
# Styling
# =========================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 6px;
            color: #0f172a;
        }
        .author-link {
            font-size: 18px;
            font-weight: 500;
            text-decoration: none;
        }
        .section-card {
            padding: 18px 20px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background-color: #ffffff;
            margin-bottom: 18px;
        }
        .small-muted {
            color: #64748b;
            font-size: 14px;
        }
        .footer {
            color: #64748b;
            font-size: 13px;
            margin-top: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Load Project Files
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    metrics = load_metrics()
    df = load_data()
    project_ready = True
except Exception as e:
    project_ready = False
    st.error("Error loading project files.")
    st.exception(e)

# =========================================================
# Helper Functions
# =========================================================
def show_bar(series, title, xlabel, ylabel, rotation=0):
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    st.pyplot(fig)


def show_line(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index, series.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def diabetes_percentage_by_binary_feature(data, feature, labels):
    percentages = pd.crosstab(
        data[feature],
        data["Diabetes_012"],
        normalize="index"
    ) * 100
    percentages.index = labels
    percentages.columns = ["No Diabetes", "Diabetes"]
    return percentages.round(2)


def diabetes_percentage_by_feature(data, feature):
    percentages = pd.crosstab(
        data[feature],
        data["Diabetes_012"],
        normalize="index"
    ) * 100
    percentages.columns = ["No Diabetes", "Diabetes"]
    return percentages.round(2)


def get_score(metric_name):
    if metrics is None or metrics.empty:
        return None
    if "Metric" not in metrics.columns or "Score" not in metrics.columns:
        return None
    selected = metrics.loc[metrics["Metric"] == metric_name, "Score"]
    if selected.empty:
        return None
    return float(selected.iloc[0])

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## Project Information")
st.sidebar.write(
    "This project predicts diabetes risk using selected health, lifestyle, "
    "and demographic features from the CDC BRFSS 2024 dataset."
)
st.sidebar.markdown("**Final Model:** Tuned Random Forest")
st.sidebar.markdown("**Model Type:** Classification")
st.sidebar.markdown("**Decision Threshold:** 50%")
st.sidebar.markdown("**Number of Features:** 13")
st.sidebar.divider()
st.sidebar.markdown("## Final Model Metrics")
if project_ready:
    st.sidebar.dataframe(metrics, use_container_width=True, hide_index=True)
else:
    st.sidebar.warning("Metrics file was not loaded.")
st.sidebar.divider()
st.sidebar.markdown("## Important Note")
st.sidebar.info("This app is for educational purposes only and should not be used as a medical diagnosis tool.")

# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="main-title">
        🩺 BRFSS Diabetes Prediction App
        <span class="author-link">
            By <a href="{LINKEDIN_URL}" target="_blank">Yousef Ismail Ahmed</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning("This app is for educational purposes only and should not be used as a medical diagnosis tool.")

if project_ready:
    st.success("Model loaded successfully! The final pipeline is ready for prediction.")
else:
    st.stop()

# =========================================================
# Main Tabs
# =========================================================
overview_tab, eda_tab, prediction_tab, performance_tab = st.tabs([
    "📌 Overview",
    "📊 EDA / Analysis",
    "🩺 Diabetes Prediction",
    "🤖 Model Performance"
])

# =========================================================
# Overview Tab
# =========================================================
with overview_tab:
    st.subheader("Project Overview")
    st.markdown(
        """
        This project uses the **CDC BRFSS 2024** health survey dataset to build a machine learning
        classification model that predicts whether a respondent is likely to belong to the diabetes class.

        The project follows a complete machine learning workflow: data loading, data cleaning,
        exploratory data analysis, preprocessing, model training, model evaluation, cross-validation,
        hyperparameter tuning, model saving, and deployment using Streamlit.
        """
    )

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Selected Features", "13")
    with col3:
        st.metric("Target Classes", "2")
    with col4:
        st.metric("Final Model", "Tuned RF")

    st.divider()
    st.subheader("Problem Statement")
    st.write(
        "The goal is to analyze health, lifestyle, and demographic factors from the BRFSS 2024 survey "
        "and build a classification model that predicts diabetes risk based on selected features."
    )

    st.subheader("Dataset Source")
    st.write("The dataset is based on the CDC Behavioral Risk Factor Surveillance System annual 2024 public-use data.")

    st.subheader("Selected Features")
    selected_features = [
        "BMI", "GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "Stroke",
        "HeartDisease", "Smoker", "PhysicalActivity", "Sex", "Age", "Education", "Income", "HeavyAlcohol"
    ]
    st.dataframe(pd.DataFrame({"Feature": selected_features}), use_container_width=True, hide_index=True)

# =========================================================
# EDA Tab
# =========================================================
with eda_tab:
    st.subheader("Exploratory Data Analysis")
    st.write(
        "This section presents the main analytical findings from the cleaned BRFSS diabetes dataset. "
        "The analysis is divided into univariate, bivariate, and multivariate analysis."
    )

    uni_tab, bi_tab, multi_tab = st.tabs([
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis"
    ])

    with uni_tab:
        st.markdown("### Univariate Analysis")
        st.write("This section analyzes each variable individually.")
        col1, col2 = st.columns(2)

        with col1:
            target_counts = df["Diabetes_012"].value_counts().sort_index()
            target_counts.index = ["No Diabetes", "Diabetes"]
            show_bar(target_counts, "Diabetes Target Distribution", "Diabetes Status", "Count")
            st.caption("The dataset is imbalanced, with non-diabetic respondents forming the majority class.")

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            df["BMI"].dropna().plot(kind="hist", bins=40, ax=ax)
            ax.set_title("BMI Distribution")
            ax.set_xlabel("BMI")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            st.caption("Most BMI values are concentrated between the normal and overweight ranges, with a right-skewed distribution.")

        col3, col4 = st.columns(2)
        with col3:
            age_counts = df["Age"].value_counts().sort_index()
            show_bar(age_counts, "Age Group Distribution", "Age Group", "Count")
        with col4:
            general_health_counts = df["GeneralHealth"].value_counts().sort_index()
            general_health_counts.index = ["Fair/Poor Health", "Good/Better Health"]
            show_bar(general_health_counts, "General Health Distribution", "General Health", "Count")

    with bi_tab:
        st.markdown("### Bivariate Analysis")
        st.write("This section compares diabetes status with key health and lifestyle factors.")
        col1, col2 = st.columns(2)

        with col1:
            bmi_by_diabetes = df.groupby("Diabetes_012")["BMI"].mean()
            bmi_by_diabetes.index = ["No Diabetes", "Diabetes"]
            show_bar(bmi_by_diabetes.round(2), "Average BMI by Diabetes Status", "Diabetes Status", "Average BMI")
            st.caption("Respondents in the diabetes class have a higher average BMI than non-diabetic respondents.")

        with col2:
            general_health_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["GeneralHealth"]),
                "GeneralHealth",
                ["Fair/Poor Health", "Good/Better Health"]
            )
            show_bar(general_health_percentages["Diabetes"], "Diabetes Percentage by General Health", "General Health", "Diabetes Percentage (%)")
            st.caption("Diabetes percentage is higher among respondents with fair or poor general health.")

        col3, col4 = st.columns(2)
        with col3:
            age_diabetes_percentages = diabetes_percentage_by_feature(df, "Age")
            show_line(age_diabetes_percentages["Diabetes"], "Diabetes Percentage by Age Group", "Age Group", "Diabetes Percentage (%)")
            st.caption("Diabetes percentage generally increases as age group increases.")
        with col4:
            physical_activity_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["PhysicalActivity"]),
                "PhysicalActivity",
                ["No Physical Activity", "Had Physical Activity"]
            )
            show_bar(physical_activity_percentages["Diabetes"], "Diabetes Percentage by Physical Activity", "Physical Activity Status", "Diabetes Percentage (%)")
            st.caption("Respondents who reported physical activity have a lower diabetes percentage.")

        col5, col6 = st.columns(2)
        with col5:
            heart_disease_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["HeartDisease"]),
                "HeartDisease",
                ["No Heart Disease", "Heart Disease"]
            )
            show_bar(heart_disease_percentages["Diabetes"], "Diabetes Percentage by Heart Disease Status", "Heart Disease Status", "Diabetes Percentage (%)")
        with col6:
            stroke_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["Stroke"]),
                "Stroke",
                ["No Stroke", "Stroke"]
            )
            show_bar(stroke_percentages["Diabetes"], "Diabetes Percentage by Stroke Status", "Stroke Status", "Diabetes Percentage (%)")

        st.markdown("### Physical and Mental Health Days")
        health_days_by_diabetes = df.groupby("Diabetes_012")[["PhysicalHealthDays", "MentalHealthDays"]].mean().round(2)
        health_days_by_diabetes.index = ["No Diabetes", "Diabetes"]
        st.dataframe(health_days_by_diabetes, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        health_days_by_diabetes.plot(kind="bar", ax=ax)
        ax.set_title("Average Physical and Mental Health Days by Diabetes Status")
        ax.set_xlabel("Diabetes Status")
        ax.set_ylabel("Average Number of Days")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        st.caption("Diabetic respondents reported higher average physically unhealthy days and slightly higher mentally unhealthy days.")

        col7, col8 = st.columns(2)
        with col7:
            smoker_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["Smoker"]),
                "Smoker",
                ["Never Smoker", "Smoker / Former Smoker"]
            )
            show_bar(smoker_percentages["Diabetes"], "Diabetes Percentage by Smoking Status", "Smoking Status", "Diabetes Percentage (%)")
        with col8:
            heavy_alcohol_percentages = diabetes_percentage_by_binary_feature(
                df.dropna(subset=["HeavyAlcohol"]),
                "HeavyAlcohol",
                ["Not Heavy Alcohol", "Heavy Alcohol"]
            )
            show_bar(heavy_alcohol_percentages["Diabetes"], "Diabetes Percentage by Heavy Alcohol Consumption", "Heavy Alcohol Status", "Diabetes Percentage (%)")

    with multi_tab:
        st.markdown("### Multivariate Analysis")
        st.write("This section explores relationships between multiple numerical variables in the dataset.")

        correlation_matrix = df.corr(numeric_only=True).round(2)
        fig, ax = plt.subplots(figsize=(10, 7))
        image = ax.imshow(correlation_matrix)
        plt.colorbar(image, ax=ax)
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_yticklabels(correlation_matrix.columns)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        st.markdown("### Top Correlations with Diabetes Target")
        diabetes_correlations = correlation_matrix["Diabetes_012"].drop("Diabetes_012").sort_values(ascending=False)
        st.dataframe(
            diabetes_correlations.reset_index().rename(columns={"index": "Feature", "Diabetes_012": "Correlation with Diabetes"}),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "The correlations are not very high, which is expected in real health survey data. "
            "However, features such as age, BMI, heart disease, physical health days, and stroke show positive relationships with diabetes status."
        )

# =========================================================
# Prediction Tab
# =========================================================
with prediction_tab:
    st.subheader("Diabetes Risk Prediction")
    st.write("Enter the respondent health information below, then click Predict.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=28.0, step=0.1)
            general_health_label = st.selectbox("General Health", ["Good / Better Health", "Fair / Poor Health"])
            physical_health_days = st.slider("Physical Health Days", min_value=0, max_value=30, value=0)
            mental_health_days = st.slider("Mental Health Days", min_value=0, max_value=30, value=0)
            stroke_label = st.selectbox("Stroke", ["No", "Yes"])
        with col2:
            heart_disease_label = st.selectbox("Heart Disease", ["No", "Yes"])
            smoker_label = st.selectbox("Smoking Status", ["Never Smoker", "Smoker / Former Smoker"])
            physical_activity_label = st.selectbox("Physical Activity", ["Had Physical Activity", "No Physical Activity"])
            sex_label = st.selectbox("Sex", ["Female", "Male"])
        with col3:
            age_label = st.selectbox(
                "Age Group",
                [
                    "18 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54",
                    "55 to 59", "60 to 64", "65 to 69", "70 to 74", "75 to 79", "80 or older", "Unknown / Refused"
                ],
                index=7
            )
            education_label = st.selectbox(
                "Education Level",
                [
                    "Did not graduate High School", "Graduated High School",
                    "Attended College / Technical School", "Graduated College / Technical School"
                ]
            )
            income_label = st.selectbox(
                "Income Level",
                [
                    "Less than $15,000", "$15,000 to < $25,000", "$25,000 to < $35,000",
                    "$35,000 to < $50,000", "$50,000 to < $100,000", "$100,000 to < $200,000",
                    "$200,000 or more"
                ],
                index=4
            )
            heavy_alcohol_label = st.selectbox("Heavy Alcohol Consumption", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Diabetes Risk")

    with st.expander("Feature Guide"):
        st.markdown(
            """
            **Age Group:** Based on BRFSS age categories.  
            **Income Level:** Based on BRFSS income categories.  
            **Physical / Mental Health Days:** Number of unhealthy days during the past 30 days.  
            **Prediction Probability:** The model's estimated probability for the diabetes class.
            """
        )

    if submitted:
        general_health = 1 if general_health_label == "Good / Better Health" else 0
        stroke = 1 if stroke_label == "Yes" else 0
        heart_disease = 1 if heart_disease_label == "Yes" else 0
        smoker = 1 if smoker_label == "Smoker / Former Smoker" else 0
        physical_activity = 1 if physical_activity_label == "Had Physical Activity" else 0
        sex = 1 if sex_label == "Male" else 0
        heavy_alcohol = 1 if heavy_alcohol_label == "Yes" else 0

        age_mapping = {
            "18 to 24": 1, "25 to 29": 2, "30 to 34": 3, "35 to 39": 4, "40 to 44": 5,
            "45 to 49": 6, "50 to 54": 7, "55 to 59": 8, "60 to 64": 9, "65 to 69": 10,
            "70 to 74": 11, "75 to 79": 12, "80 or older": 13, "Unknown / Refused": 14
        }
        education_mapping = {
            "Did not graduate High School": 1,
            "Graduated High School": 2,
            "Attended College / Technical School": 3,
            "Graduated College / Technical School": 4
        }
        income_mapping = {
            "Less than $15,000": 1,
            "$15,000 to < $25,000": 2,
            "$25,000 to < $35,000": 3,
            "$35,000 to < $50,000": 4,
            "$50,000 to < $100,000": 5,
            "$100,000 to < $200,000": 6,
            "$200,000 or more": 7
        }

        input_data = pd.DataFrame({
            "BMI": [bmi],
            "GeneralHealth": [general_health],
            "PhysicalHealthDays": [physical_health_days],
            "MentalHealthDays": [mental_health_days],
            "Stroke": [stroke],
            "HeartDisease": [heart_disease],
            "Smoker": [smoker],
            "PhysicalActivity": [physical_activity],
            "Sex": [sex],
            "Age": [age_mapping[age_label]],
            "Education": [education_mapping[education_label]],
            "Income": [income_mapping[income_label]],
            "HeavyAlcohol": [heavy_alcohol]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percentage = probability * 100

        st.divider()
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("Prediction: Higher Diabetes Risk")
        else:
            st.success("Prediction: Lower Diabetes Risk")

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

# =========================================================
# Performance Tab
# =========================================================
with performance_tab:
    st.subheader("Final Model Performance")
    st.write(
        "The final selected model is a **Tuned Random Forest** pipeline. "
        "The saved pipeline contains both preprocessing steps and the classification model."
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.divider()
    accuracy = get_score("Accuracy")
    precision = get_score("Precision")
    recall = get_score("Recall")
    f1 = get_score("F1-score")
    roc_auc = get_score("ROC-AUC")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}" if accuracy is not None else "N/A")
    with col2:
        st.metric("Precision", f"{precision:.4f}" if precision is not None else "N/A")
    with col3:
        st.metric("Recall", f"{recall:.4f}" if recall is not None else "N/A")
    with col4:
        st.metric("F1-score", f"{f1:.4f}" if f1 is not None else "N/A")
    with col5:
        st.metric("ROC-AUC", f"{roc_auc:.4f}" if roc_auc is not None else "N/A")

    st.divider()
    st.subheader("Why Recall Matters")
    st.write(
        "This is an imbalanced healthcare-related classification problem. "
        "Recall is important because missing actual diabetic cases may be more serious than producing false positives."
    )

    st.subheader("Final Model Selection")
    st.write(
        "The Tuned Random Forest model was selected as the final model because it achieved the best overall "
        "recall and ROC-AUC among the tested models while keeping the workflow deployable through a saved pipeline."
    )

# =========================================================
# Footer
# =========================================================
st.divider()
st.markdown(
    """
    <div class="footer">
        BRFSS Diabetes Prediction Project | Built with Python, Scikit-learn, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
