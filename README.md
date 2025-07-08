# Credit-Risk-Modeling
# 🧠 Credit Risk Modeling – AMEX Case Study

An end-to-end machine learning project to predict credit default risk using American Express’s real-world dataset from the [Kaggle AMEX Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction) competition.

Developed by **Pranay Kumar Reddy Yammanuru**, this solution tackles a large-scale credit risk modeling challenge with explainability, efficiency, and business relevance in mind.

---

## 🧭 Project Overview

In today’s evolving financial landscape, predicting customer creditworthiness is critical for minimizing defaults and enhancing customer experience. This project leverages machine learning techniques to:

- Identify high-risk customers using time-series behavioral and transactional data
- Improve decision-making for lending and credit approval
- Enhance transparency with model explainability (SHAP)

---

## 🎯 Objective

To build a production-ready and explainable credit risk model that can rival existing systems used by major financial institutions. Two models were developed and evaluated:

- ✅ **XGBoost** (Best performer)
- 🧪 **Neural Network**

---

## 🔗 Data Source

- **Kaggle**: [AMEX Default Prediction Competition](https://www.kaggle.com/competitions/amex-default-prediction)
- **Rows**: 5.5M+  
- **Customers**: 91,783  
- **Time Span**: 13 months  

---

## ⚙️ Feature Engineering

- Aggregated temporal features: `mean`, `min`, `max`, `last` over 12 months
- One-hot encoded 11 categorical columns → 45+ dummy variables
- Missing value treatment:
  - Binary → `0`
  - Categorical → `Mode`
  - Numeric → `Mean`
- Created domain-specific variable groupings:
  - `D_` – Delinquency  
  - `B_` – Balance  
  - `S_` – Spend  
  - `P_` – Payment  
  - `R_` – Risk  

---

## 🧠 Key Highlights

- Efficiently handled a large-scale dataset using chunk processing
- Built 72 XGBoost models with exhaustive grid search
- Created 32 Neural Network variations with different layer depths, activation functions, and batch sizes
- Used **SHAP** values to explain predictions and feature contributions

---

## 📊 Model Training & Evaluation

### ✅ XGBoost
- Grid search on:
  - Trees: 50, 100, 300  
  - Learning rate: 0.01, 0.1  
  - Subsample: 50%, 80%  
  - Feature sample: 50%, 100%  
  - Class weight: 1, 5, 10  
- **Best AUC** on Test Set 2  
- **SHAP Analysis** revealed key predictors like `P_2_last`, `B_1_last`, `D_39_max`

### 🧪 Neural Network
- Applied standard scaling and outlier capping at the 99th percentile
- Grid search on:
  - Layers: 2, 4  
  - Neurons: 4, 6  
  - Activation: ReLU, Tanh  
  - Dropout: 50%, 100%  
  - Batch size: 100, 10,000  
- Underperformed compared to XGBoost

---

## 🧮 SHAP Value Insights (XGBoost)

| Feature       | Impact    |
|---------------|-----------|
| P_2_last      | Negative  |
| B_1_last      | Negative  |
| D_39_max      | Positive  |
| S_3_mean      | Positive  |

---

## 🏆 Final Model Selection

| Model        | AUC (Test 2) | Remarks            |
|--------------|--------------|--------------------|
| XGBoost      | ✅ High       | Final Model        |
| Neural Net   | Moderate     | Not selected       |

- **XGBoost** selected for its strong AUC and stability across training, validation, and test splits

---

## 🚀 Future Enhancements

- 📊 Streamlit dashboard for real-time model scoring  
- 🤖 AutoML comparison with additional algorithms  
- 🛡️ Model drift detection for production monitoring  

---
## 📁 Project Structure

📦 Credit_Risk_Modeling
┣ 📊 AMEX_Credit_Risk_Analysis.ipynb
┣ 🧠 credit_risk_modeling.py
┣ 📄 final_data_for_project.csv
┣ 📈 Credit Risk Model.pptx
┗ 📘 README.md

