# Student Performance Prediction & Classification Analysis

A machine learning project exploring student performance dataset to predict final Mathematics grades (`G3.Math`) using Regression models and multi-class target binning with Classification models.

---

## 📌 Project Overview

This repository analyzes student demographic, social, and academic factors to predict student academic outcomes. The dataset combines student data across Mathematics and Portuguese language subjects. 

The primary goal is to evaluate and compare linear, tree-based, and distance-based machine learning algorithms under both **Regression** (predicting exact grades) and **Classification** (predicting binned performance brackets) setups.

---

## 🛠️ Data Preprocessing & Pipeline Steps

1. **Feature Dropping:** Dropped irrelevant identifiers, `Unnamed` columns, and duplicated domain metrics.
2. **One-Hot Encoding:** Converted categorical variables into numerical indicators using `pd.get_dummies()`.
3. **Correlation Analysis:** Evaluated feature relationships using Seaborn heatmaps to inspect linear dependencies.
4. **Feature Importance Selection:** Identified key predictive drivers using tree-based feature importances.
5. **Target Binning:** Quantile-binned the target variable `G3.Math` into 5 performance groups (`q=5`) for classification tasks.

---

## 📊 Model Performance & Results

### 1. Regression Models (Target: Continuous `G3.Math`)
Predicting exact final grades on a continuous scale:

| Model | $R^2$ Score | MAE | MSE | Cross-Validation ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 0.20 | 2.65 | 12.87 | ~0.19 |
| **Random Forest Regressor** | **0.80** | **1.07** | **3.19** | **~0.73** |

* **Key Takeaway:** Linear models struggled with the non-linear feature relationships, while **Random Forest Regressor** demonstrated strong predictive accuracy ($R^2 = 0.80$).

### 2. Classification Models (Target: 5-Class Binned Grades)
Predicting student grade performance tiers (0 to 4):

| Model | Accuracy | Weighted F1-Score | Cross-Validation Score |
| :--- | :---: | :---: | :---: |
| **Support Vector Classifier (SVC)** | 34% | 0.24 | ~32.4% |
| **Random Forest Classifier** | **80%** | **0.80** | **~85.8%** |

* **Key Takeaway:** Unscaled input features caused SVC to underperform significantly, while **Random Forest Classifier** achieved over **85% mean CV accuracy** (optimizing further with `n_estimators=90/100`).

---

## 📁 Repository Structure

```text
.
├── exam_data.csv            # Raw dataset
├── student_analysis.ipynb   # Main Jupyter / Google Colab notebook
└── README.md                # Project documentation
