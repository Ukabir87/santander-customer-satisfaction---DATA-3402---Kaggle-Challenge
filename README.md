<p align="center">
  <img src="UTA_DataScience_Logo.png" width="200"/>
</p>

# Santander Customer Satisfaction Prediction

This project uses machine learning to predict whether a Santander customer is satisfied or dissatisfied based on anonymized financial and behavioral data. The task is a binary classification problem using a Kaggle tabular dataset.

---

## Overview

The goal of this project is to identify customers who are likely to be dissatisfied using supervised machine learning. The dataset contains over 300 anonymized numerical features and a binary target variable (`TARGET`).

A key challenge in this dataset is **class imbalance**, where the majority of customers are satisfied (class 0) and only a small portion are unsatisfied (class 1). Because of this, **ROC-AUC** was used as the primary evaluation metric instead of accuracy.

Multiple models were trained and compared, and the best-performing model was selected based on validation performance.

---

## Summary of Work Done

### Data

- **Type:** CSV files from Kaggle
- **Input:** 300+ anonymized numerical features
- **Output:** Binary label (`TARGET`)
  - 0 → Satisfied
  - 1 → Unsatisfied
- **Dataset Size:** ~76,000 rows (train)
- **Split:**
  - 80% training
  - 20% validation
  - Stratified split to maintain class distribution

---

### Preprocessing / Cleanup

- Removed constant columns (features with no variation)
- Removed duplicate columns to eliminate redundant information
- Fixed abnormal values in `var3` (`-999999` → replaced with 2)
- Dropped `ID` column since it has no predictive value
- Separated features and target variables
- Stored test IDs for submission

---

### Problem Formulation

This is a **binary classification problem**:

- **Target Variable:** `TARGET`
- **Classes:**
  - 0 → Satisfied
  - 1 → Unsatisfied

- **Evaluation Metric:** ROC-AUC
- **Reason:** Dataset is imbalanced, so accuracy is misleading

---

### Models Used

The following models were trained and compared:

- Logistic Regression (baseline model)
- Random Forest Classifier
- Extra Trees Classifier
- Histogram Gradient Boosting (best performing model)

---

### Training

Model training was performed using Python and scikit-learn in a Jupyter Notebook.

- Each model was trained on the training set
- Predictions were made using `predict_proba()` to obtain probability scores
- ROC-AUC was calculated on the validation set
- Results were stored and compared to select the best model

---

### Performance Comparison

| Model | Validation ROC-AUC |
|------|------------------|
| Histogram Gradient Boosting | ~0.847 |
| Random Forest | ~0.821 |
| Logistic Regression | ~0.803 |
| Extra Trees | ~0.790 |

The **Histogram Gradient Boosting model** achieved the highest ROC-AUC score and was selected as the final model.

---

### Final Output

- Best model retrained on full dataset
- Predictions generated on test data using probability scores
- Final submission file created:
  - `santander_submission.csv`

---

### Key Observations

- Ensemble methods outperformed simpler models
- Data cleaning significantly improved performance
- ROC-AUC provided a better evaluation than accuracy
- The model tends to favor the majority class due to imbalance

---

### Future Improvements

- Feature engineering (e.g., interaction features)
- Use advanced models like XGBoost or LightGBM
- Apply class balancing techniques (SMOTE, etc.)

---

## Visualizations

Here are some of the visualizations created throughout the notebook to better understand the data and how the models performed.

---

### Feature Distribution Histograms

Density histograms were plotted for a few key features to get a feel for how the data is spread out. Density was used instead of raw counts so the plots are easier to compare, and the x-axis was trimmed to cut off extreme outliers. A lot of the features turned out to be heavily skewed toward zero.

| var3 | var15 |
|------|-------|
| ![feature_dist_var3](images/feature_dist_var3.png) | ![feature_dist_var15](images/feature_dist_var15.png) |

| var36 | var38 |
|-------|-------|
| ![feature_dist_var36](images/feature_dist_var36.png) | ![feature_dist_var38](images/feature_dist_var38.png) |

| num_var4 |
|----------|
| ![feature_dist_num_var4](images/feature_dist_num_var4.png) |

---

### Class-Based Density Histograms

Same idea as above, but this time the distributions for satisfied vs. unsatisfied customers are overlaid on the same plot. Using density here was important because there are way fewer unsatisfied customers, so raw counts would make class 1 nearly invisible. This makes it easier to spot which features actually look different between the two groups.

| var3 | var15 |
|------|-------|
| ![class_density_var3](images/class_density_var3.png) | ![class_density_var15](images/class_density_var15.png) |

| var36 | var38 |
|-------|-------|
| ![class_density_var36](images/class_density_var36.png) | ![class_density_var38](images/class_density_var38.png) |

| num_var4 |
|----------|
| ![class_density_num_var4](images/class_density_num_var4.png) |

---

### ROC Curve for Best Model

The ROC curve for the best model shows how well it separates the two classes. The dashed line represents random guessing — the further the curve bows away from it, the better. The final model landed at an AUC of **0.889**.

![ROC Curve](images/roc_curve.png)

---

### Threshold Tuning Visualization

The default 0.5 threshold doesn't work great when classes are imbalanced, so this plot shows how precision, recall, and F1 change as the threshold shifts. It helps find a cutoff that does a better job of actually catching unsatisfied customers.

![Threshold Tuning](images/threshold_tuning.png)

---

## Repository Structure
```
Santander-Customer-Satisfaction/
│
├── Final Notebook/
│ └── santander_customer_satisfaction_complete.ipynb
│
├── dataset/
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
│
├── santander_submission.csv
├── README.md
```

---

## How to Run the Project

1. Clone the repository  

2. Run the notebook from Final Notebook Folder:
   - `santander_customer_satisfaction_complete.ipynb`
---

## Required Libraries

- pandas
- numpy
- scikit-learn

---

## Citations

- [Kaggle Competition — Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction/overview)
