# 💳 Credit Card Fraud Detection System

This project was developed during the **Infosys Springboard Internship** by **Shailesh K R** under the guidance of **Mr. M. Muvendiran**. The objective is to detect fraudulent credit card transactions using machine learning models with a focus on high accuracy, reliability, and financial security.

---

## 📌 Overview

Fraudulent transactions pose a significant threat to financial institutions and customers. This project aims to identify such anomalies in transaction data using advanced ML techniques.

### 🎯 Objectives

- Detect fraudulent credit card transactions with high precision.
- Minimize false positives and negatives.
- Visualize data insights using EDA techniques.
- Evaluate various machine learning algorithms to identify the most effective one.

---

## 📂 Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (~0.17%)
- **Challenge**: Highly imbalanced dataset.

---

## 📊 Data Exploration & Preprocessing

- Handled null/missing values.
- Addressed class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
- Scaled features using `StandardScaler`.
- Performed EDA using correlation matrices, box plots, and heatmaps.

---

## 🧠 Machine Learning Models Used

| Model              | Highlights                                                              |
|-------------------|--------------------------------------------------------------------------|
| Random Forest      | Robust ensemble model with good generalization capabilities             |
| Logistic Regression| Interpretable and efficient for binary classification                   |
| Decision Tree      | Simple, intuitive, and visually interpretable                           |
| Support Vector Machine (SVM) | Effective with non-linear boundaries using kernel tricks     |
| XGBoost            | High-performing gradient boosting model; best results in this project   |

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC Curve**

---

## ✅ Results

- **Best Performing Model**: `XGBoost`
- **Accuracy Achieved**: **96%**
- Confusion Matrix and classification reports indicate minimal false positives/negatives.
- Key Features: Transaction **Amount** and **Time-based patterns** are highly indicative of fraud.

---

## 🚀 Future Scope

- Deploy the fraud detection model into real-time transaction systems.
- Enhance performance further with **Deep Learning** (e.g., LSTM, Autoencoders).
- Integrate with APIs and frontend interfaces for live monitoring.

---

## 🙏 Acknowledgments

- **Infosys Springboard Internship Program** for the opportunity and mentorship.
- **Kaggle** for providing the dataset.

---

## 📁 Repository Structure

