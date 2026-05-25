![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed%20App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange)
![EDA](https://img.shields.io/badge/EDA-Analysis-purple)

# 🩺 BRFSS Diabetes Prediction App

## 📌 Project Overview

This project builds a **machine learning classification model** to predict diabetes risk using selected health, lifestyle, and demographic features from the **CDC BRFSS 2024 dataset**.

The project follows a complete data science workflow starting from understanding the dataset source, selecting relevant variables, cleaning and preprocessing the data, performing exploratory data analysis, training and comparing machine learning models, tuning the final model, saving the trained pipeline, and deploying the final application using Streamlit.

The final deployed model is a **Tuned Random Forest pipeline** that includes both preprocessing and prediction steps.

---

## 🌐 Live Links

- **Streamlit App:** https://brfss-diabetes-prediction.streamlit.app/
- **GitHub Repository:** https://github.com/Youssef-I-Ahmed/BRFSS-Diabetes-Prediction
- **LinkedIn:** https://www.linkedin.com/in/yousef-ismail87/

---

## 🏥 Dataset Source

The dataset used in this project comes from the **Behavioral Risk Factor Surveillance System (BRFSS)**, which is provided by the **CDC — Centers for Disease Control and Prevention**.

BRFSS is a large annual health survey conducted in the United States. It collects information about health conditions, lifestyle habits, demographics, and risk factors.

For this project, the latest available **BRFSS 2024** data was used.

### Dataset Links

- Kaggle dataset reference:  
  [https://www.kaggle.com/datasets/rudritarahman/cdc-brfss-survey-data-2024](https://www.kaggle.com/datasets/rudritarahman/cdc-brfss-survey-data-2024)

- Official CDC BRFSS annual data page:  
  https://www.cdc.gov/brfss/annual_data/annual_2024.html

- Raw data file used:  
  `2024 BRFSS Data (SAS Transport Format)`

- Codebook used to understand the variables:  
  `2024 BRFSS Codebook CDC`

---

## 🎯 Problem Statement

The goal of this project is to analyze selected health, lifestyle, and demographic factors from the CDC BRFSS 2024 survey and build a machine learning model that predicts whether a respondent is likely to have diabetes or not.

This is a **binary classification problem**:

- `0` → No Diabetes
- `1` → Diabetes

> Important: This project is for educational purposes only and should not be used as a medical diagnosis tool.

---

## 🧾 Selected Features

The original BRFSS dataset contains hundreds of columns. To make the project focused and meaningful, selected variables related to diabetes risk were used.

| Original Column | Final Feature Name | Description |
|---|---|---|
| `DIABETE4` | `Diabetes_012` | Diabetes target variable |
| `_BMI5` | `BMI` | Body Mass Index |
| `_RFHLTH` | `GeneralHealth` | General health status |
| `PHYSHLTH` | `PhysicalHealthDays` | Number of physically unhealthy days in the past 30 days |
| `MENTHLTH` | `MentalHealthDays` | Number of mentally unhealthy days in the past 30 days |
| `CVDSTRK3` | `Stroke` | Stroke history |
| `_MICHD` | `HeartDisease` | Heart disease history |
| `_SMOKER3` | `Smoker` | Smoking status |
| `_TOTINDA` | `PhysicalActivity` | Physical activity status |
| `SEXVAR` | `Sex` | Respondent sex |
| `_AGEG5YR` | `Age` | Age group |
| `_EDUCAG` | `Education` | Education level |
| `_INCOMG1` | `Income` | Income group |
| `_RFDRHV9` | `HeavyAlcohol` | Heavy alcohol consumption |

---

## 🧹 Data Cleaning Summary

The raw BRFSS dataset was cleaned and transformed before modeling.

Main cleaning steps included:

- Selecting only the relevant columns needed for the project.
- Renaming columns to more readable names.
- Cleaning the target variable `DIABETE4`.
- Converting the problem into binary classification:
  - Diabetes cases were mapped to `1`.
  - No diabetes and prediabetes/borderline cases were mapped to `0`.
  - Unknown, refused, and invalid responses were removed.
- Converting BMI from BRFSS format:
  - BRFSS stores BMI as `BMI * 100`.
  - Example: `2700` becomes `27.00`.
- Replacing special survey values such as `77`, `88`, and `99` where needed.
- Handling invalid and missing values.
- Saving the cleaned dataset as:

```text
data/brfss_diabetes_cleaned.csv
```

---

## 🔍 Exploratory Data Analysis

The EDA phase was divided into three main parts:

### 1. Univariate Analysis

This part focused on understanding each variable separately.

Examples:

- Target distribution
- BMI distribution
- Age group distribution
- General health distribution
- Physical and mental health days distribution

Key finding:

- The target variable is imbalanced.
- Most respondents are in the no-diabetes class.

### 2. Bivariate Analysis

This part focused on comparing individual features with the diabetes target.

Examples:

- Diabetes percentage by BMI
- Diabetes percentage by age group
- Diabetes percentage by general health
- Diabetes percentage by physical activity
- Diabetes percentage by smoking status
- Diabetes percentage by heart disease and stroke

Key findings:

- Diabetes percentage generally increases with age.
- Respondents with higher BMI tend to have higher diabetes risk.
- Poor general health is strongly associated with diabetes.
- Respondents with heart disease or stroke have higher diabetes percentages.
- Respondents with no physical activity show higher diabetes percentages.

### 3. Multivariate Analysis

This part focused on relationships between multiple variables.

Examples:

- Correlation heatmap
- Comparing numerical and binary features together
- Understanding which features have stronger relationships with diabetes

Key findings:

- Age, BMI, heart disease, stroke, and physical health days showed positive relationships with diabetes.
- The correlation values were useful for understanding relationships, but model performance was evaluated using classification metrics.

---

## 🤖 Machine Learning Workflow

The machine learning pipeline included:

1. Splitting the data into training and testing sets.
2. Using stratified split to preserve the target distribution.
3. Building a preprocessing pipeline.
4. Handling missing values using median imputation.
5. Scaling numerical features using StandardScaler.
6. Training Logistic Regression.
7. Training Random Forest.
8. Comparing model performance.
9. Applying Cross Validation.
10. Performing hyperparameter tuning using RandomizedSearchCV.
11. Selecting the final tuned Random Forest model.
12. Saving the final pipeline using Joblib.

---

## 🧠 Models Used

The following models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **Tuned Random Forest**

Because the target variable is imbalanced, accuracy alone was not enough. The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 📊 Final Model Performance

The final selected model is the **Tuned Random Forest pipeline**.

| Metric | Score |
|---|---:|
| Accuracy | 0.7004 |
| Precision | 0.2891 |
| Recall | 0.7291 |
| F1-score | 0.4140 |
| ROC-AUC | 0.7866 |

The final model was selected mainly because it achieved a good balance between recall and ROC-AUC.

In this healthcare-related classification problem, recall is important because missing actual diabetic cases may be more serious than producing false positives.

---

## 🚀 Streamlit Application

The project includes a deployed Streamlit application that allows users to enter health and demographic information and receive a diabetes risk prediction.

The app includes:

- Project overview
- EDA / Analysis section
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Diabetes risk prediction form
- Prediction result
- Diabetes probability
- Probability progress bar
- Final model metrics
- Feature guide
- Educational disclaimer

---

## 🗂️ Project Structure

```text
BRFSS-Diabetes-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── final_check.py
├── .gitignore
│
├── data/
│   ├── LLCP2024.XPT
│   └── brfss_diabetes_cleaned.csv
│
├── models/
│   ├── diabetes_rf_pipeline.pkl
│   └── final_model_metrics.csv
│
└── notebooks/
    ├── 01_brfss_diabetes_analysis.ipynb
    ├── 02_eda_analysis.ipynb
    ├── 03_modeling_pipeline.ipynb
    └── 04_model_evaluation_saving.ipynb
```

---

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Youssef-I-Ahmed/BRFSS-Diabetes-Prediction.git
cd BRFSS-Diabetes-Prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal:

```text
http://localhost:8501
```

---

## ✅ Final Run Check

A final project check was performed using `final_check.py`.

The final check confirmed that:

- Required files exist.
- Required packages can be imported.
- The saved model can be loaded successfully.
- The saved metrics file can be loaded successfully.
- The final pipeline can make predictions.
- The project is ready for submission and deployment.

---

## 🧰 Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit
- Git & GitHub

---

## ⚠️ Important Disclaimer

This application is for **educational purposes only**.

The prediction result is based on a machine learning model trained on survey data and should not be used as a medical diagnosis. For any medical concerns, users should consult a qualified healthcare professional.

---

## 👤 Author

**Yousef Ismail Ahmed**  
Data Science & AI-Based Software Development Trainee

🔗 LinkedIn:  
https://www.linkedin.com/in/yousef-ismail87/
