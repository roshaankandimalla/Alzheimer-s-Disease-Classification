# Alzheimer's Disease Classification

A machine learning project for classifying Alzheimer's disease diagnosis using patient demographic, lifestyle, clinical, cognitive, and functional health data.

This project includes exploratory data analysis, preprocessing, model training, model comparison, and evaluation using multiple supervised machine learning algorithms.

## Project Overview

Alzheimer's disease is a progressive neurological disorder that affects memory, thinking, and daily functioning. This project uses structured patient data to build classification models that predict whether a patient is diagnosed with Alzheimer's disease.

The goal is to compare multiple machine learning models and identify the best-performing approach based on accuracy, precision, recall, F1-score, confusion matrices, ROC curves, and cross-validation.

> Note: This project is for educational and portfolio purposes only. It is not intended for medical diagnosis or clinical decision-making.

## Repository Contents

```text
Alzheimer-s-Disease-Classification/
|
|-- Alzheimer's_ (1).ipynb
|-- Alzheimer's_models.ipynb
|-- alzheimers_disease_data (1).csv
`-- README.md
```

## Dataset

The dataset contains **2,149 patient records** and **35 columns** related to Alzheimer's disease risk factors and diagnosis.

### Target Variable

- `Diagnosis`
  - `0`: No Alzheimer's diagnosis
  - `1`: Alzheimer's diagnosis

### Target Distribution

| Class | Count | Percentage |
|---|---:|---:|
| No Alzheimer's | 1,389 | 64.63% |
| Alzheimer's | 760 | 35.37% |

### Feature Categories

The dataset includes:

- Demographic features: `Age`, `Gender`, `Ethnicity`, `EducationLevel`
- Lifestyle features: `BMI`, `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`, `SleepQuality`
- Medical history: `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension`
- Clinical measurements: `SystolicBP`, `DiastolicBP`, cholesterol-related values
- Cognitive and functional scores: `MMSE`, `FunctionalAssessment`, `ADL`
- Symptoms and behavioral indicators: `MemoryComplaints`, `BehavioralProblems`, `Confusion`, `Disorientation`, `Forgetfulness`

## Notebook Workflow

### 1. Exploratory Data Analysis

File: `Alzheimer's_ (1).ipynb`

This notebook covers:

- Dataset loading and inspection
- Column names, data types, and missing value checks
- Descriptive statistics
- Diagnosis class distribution
- Univariate analysis for numerical and categorical variables
- Bivariate analysis against diagnosis
- Outlier analysis
- Correlation analysis
- Feature preprocessing using:
  - Standard scaling
  - One-hot encoding

### 2. Model Training and Evaluation

File: `Alzheimer's_models.ipynb`

This notebook covers:

- Train-test split
- Feature-target separation
- Model training
- Model comparison
- Confusion matrix visualization
- ROC curve analysis
- Classification reports
- Cross-validation
- Feature importance analysis

## Machine Learning Models Used

The following models were trained and compared:

- Logistic Regression
- Support Vector Machine
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| LightGBM | 0.9512 | 0.9510 | 0.9512 | 0.9511 |
| Gradient Boosting | 0.9465 | 0.9464 | 0.9465 | 0.9465 |
| Random Forest | 0.9442 | 0.9447 | 0.9442 | 0.9436 |
| XGBoost | 0.9419 | 0.9417 | 0.9419 | 0.9417 |
| SVM | 0.8535 | 0.8528 | 0.8535 | 0.8504 |
| Logistic Regression | 0.8163 | 0.8155 | 0.8163 | 0.8159 |

The best-performing model in this project was **LightGBM**, with a testing accuracy of approximately **95.12%**.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- LightGBM
- Joblib
- Jupyter Notebook

## How to Run This Project

1. Clone the repository:

```bash
git clone https://github.com/roshaankandimalla/Alzheimer-s-Disease-Classification.git
cd Alzheimer-s-Disease-Classification
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib notebook
```

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open and run the notebooks in order:

```text
1. Alzheimer's_ (1).ipynb
2. Alzheimer's_models.ipynb
```

## Key Insights

- The dataset contains no missing values.
- The target variable is moderately imbalanced, with more non-Alzheimer's cases than Alzheimer's cases.
- Ensemble models performed better than linear and kernel-based models.
- LightGBM achieved the highest overall performance.
- Cognitive and functional assessment features are important for classification.

## Future Improvements

- Add hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Handle class imbalance with SMOTE or class weighting
- Save the best model as a reusable `.pkl` or `.joblib` file
- Build a Streamlit web app for prediction
- Add model explainability using SHAP or LIME
- Add a `requirements.txt` file for easier setup

## Author

**Roshaankandimalla**

GitHub: [roshaankandimalla](https://github.com/roshaankandimalla)

## Disclaimer

This project is intended for learning and machine learning experimentation only. The predictions should not be used as medical advice, diagnosis, or treatment guidance.
