# BRFSS Diabetes Prediction App

## Project Overview

This project builds a machine learning classification model to predict diabetes risk using selected health, lifestyle, and demographic features from the BRFSS 2024 dataset.

The project follows a complete machine learning workflow, including data loading, data cleaning, exploratory data analysis, preprocessing, model training, model comparison, cross-validation, hyperparameter tuning, model saving, and Streamlit deployment.

The final deployed model is a tuned Random Forest pipeline.

---

## Dataset

The dataset is based on the Behavioral Risk Factor Surveillance System (BRFSS) 2024 data.

The original raw data contains many survey columns. For this project, selected features related to diabetes risk were used, including:

- BMI
- General health
- Physical health days
- Mental health days
- Stroke
- Heart disease
- Smoking status
- Physical activity
- Sex
- Age group
- Education
- Income
- Heavy alcohol consumption

The target variable is:

- `0`: No Diabetes
- `1`: Diabetes

---

## Project Structure

```text
BRFSS-Diabetes-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
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

## Exploratory Data Analysis Summary

The exploratory data analysis showed several important patterns related to diabetes:

- The target variable is imbalanced.
- Diabetes percentage increases with age.
- Diabetic respondents tend to have higher BMI.
- Poor general health is strongly associated with diabetes.
- Respondents with heart disease or stroke have higher diabetes percentages.
- Respondents with no physical activity have higher diabetes percentages.
- Diabetic respondents report more physically unhealthy days on average.
- Smoking history showed a higher diabetes percentage among smokers or former smokers compared to never smokers.
- The correlation heatmap showed that age, BMI, heart disease, stroke, and physical health days have positive relationships with diabetes.

---

## Models Used

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

Because the target variable is imbalanced, accuracy alone was not enough to evaluate model performance.

The main evaluation metrics were:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Final Model Performance

The final selected model is the Tuned Random Forest pipeline.

| Metric | Score |
|---|---:|
| Accuracy | 0.7004 |
| Precision | 0.2891 |
| Recall | 0.7291 |
| F1-score | 0.4140 |
| ROC-AUC | 0.7866 |

The final model was selected mainly because it achieved the best recall and ROC-AUC score among the tested models.

In this healthcare-related classification problem, recall is especially important because missing actual diabetic cases may be more serious than producing false positives.

---

## Streamlit App

The project includes a Streamlit web app that allows users to enter health and demographic information and receive a diabetes risk prediction.

The app includes:

- Prediction result
- Diabetes probability
- Probability progress bar
- Final model metrics
- Feature guide
- Project summary
- Educational disclaimer

---

## How to Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Important Note

This project is for educational purposes only.

The prediction result should not be used as a medical diagnosis. For any medical concerns, users should consult a qualified healthcare professional.

---

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

---

## Author

Yousef Ismail Ahmed
