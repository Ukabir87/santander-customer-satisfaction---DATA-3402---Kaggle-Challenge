# Santander Customer Satisfaction Prediction

## 📌 Project Overview
This project aims to predict whether a Santander customer is satisfied or dissatisfied using anonymized banking data. The dataset is taken from a Kaggle competition, and the problem is approached as a binary classification task.

The goal is to identify customers who are likely to be unhappy based on their feature values.

---

## 🎯 Objective
- Build multiple machine learning models
- Compare their performance
- Select the best model based on ROC-AUC
- Generate predictions for unseen test data

---

## 📊 Dataset
- Source: Kaggle Santander Customer Satisfaction Competition  
- Link: https://www.kaggle.com/competitions/santander-customer-satisfaction/data  

### Target Variable
- `0` → Satisfied customer  
- `1` → Unsatisfied customer  

### Important Notes
- The dataset contains 300+ anonymized numerical features
- The dataset is **imbalanced** (more satisfied customers than unsatisfied)
- Raw dataset files are not included due to large size

---

## ⚙️ Project Workflow

### 1. Data Loading
- Loaded `train.csv`, `test.csv`, and `sample_submission.csv`
- Inspected dataset shape and structure

---

### 2. Data Preprocessing
- Removed constant columns (no variation)
- Removed duplicate columns
- Fixed abnormal values in `var3` (`-999999 → 2`)
- Separated features and target
- Stored test IDs for final submission

---

### 3. Train-Test Split
- Split dataset into training and validation sets (80/20)
- Used stratification to maintain class distribution

---

### 4. Models Used
- Logistic Regression (baseline)
- Random Forest
- Extra Trees Classifier
- Histogram Gradient Boosting

---

### 5. Evaluation Metric
- **ROC-AUC Score**
- Chosen because the dataset is imbalanced
- Measures how well the model distinguishes between classes

---

## 📈 Results

| Model | Validation ROC-AUC |
|------|------------------|
| Histogram Gradient Boosting | 0.84735 |
| Random Forest | 0.82082 |
| Logistic Regression | 0.80315 |
| Extra Trees | 0.78971 |

👉 Histogram Gradient Boosting performed the best.

---

## 📤 Final Output
- Best model retrained on full dataset
- Predictions generated using `predict_proba()`
- Final submission file created:
  - `santander_submission.csv`

---

## 🧠 Key Learnings
- ROC-AUC is better than accuracy for imbalanced datasets
- Removing useless features improves model performance
- Ensemble models outperform simple models
- Proper preprocessing is critical for good results

---

## 👤 Author

**Umayer Kabir**

- Year: 2nd  
- Major: Data Science (Math Concentration)  
- University: University of Texas at Arlington

---

## 📌 Notes
This project was completed as part of a DATA-3402-PYTHON FOR DATA SCIENCE 2 course assignment. The focus was on understanding the full machine learning pipeline, from data cleaning to model evaluation and prediction.
