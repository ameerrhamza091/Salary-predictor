# 💼 Enterprise Salary Intelligence Platform

An end-to-end Machine Learning web application designed to predict market-competitive employee compensation packages based on demographic, educational, and professional parameters.

**Author:** Ameer Hamza  
**Technology Stack:** Python, Scikit-Learn, Streamlit, Pandas, NumPy, Joblib  


## 🌐 Live Demo

Aap is app ko live browser par try kar sakte hain:
👉 **[Live Demo: Enterprise Salary Predictor](https://salary-predictor-2zwgp7mddarnbvxa4vadrc.streamlit.app/)**

---


## 📌 Project Overview

Determining accurate employee compensation is critical for HR management and competitive hiring. This project analyzes employee profile datasets and leverages advanced ensemble machine learning algorithms to estimate annual salaries with high accuracy. 

The interactive web interface allows HR teams, recruiters, and candidates to evaluate predicted compensation metrics along with dynamic salary growth trajectory projections.

---

## 📊 Model Performance & Leaderboard

Multiple regression algorithms were evaluated during experimentation to select the optimal model.

| Algorithm | R² Score | RMSE Error | Scaling Required? | Final Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **~89.5%** | **~$19,417** | ❌ No | **Selected Winner** |
| Support Vector Regressor (SVM) | Evaluated | Evaluated | ✅ Yes | Distance-based |
| Random Forest Regressor | ~88.8% | ~$20,100 | ❌ No | Runner-up |
| Linear Regression | ~84.6% | ~$23,500 | ❌ No | Baseline |

> **Key Insight:** **Gradient Boosting Regressor** was selected as the deployment model due to its high stability, robust generalization on unseen data, and ability to handle raw numerical scales effectively.

---

## 🔑 Key Features

* **Real-time Prediction:** Predicts estimated annual and monthly salary packages based on inputs.
* **Feature Importance Analysis:** Identifies `Years of Experience` (~66%) as the primary driver of compensation.
* **Interactive Growth Analytics:** Renders a dynamic salary trajectory chart across 0–30 years of experience.
* **Custom UI/UX:** Styled using Streamlit with responsive components, metric badges, and dark themes.

---

## 📁 Repository Structure

```text
├── app.py                  # Streamlit web application script
├── salary_gb_model.pkl     # Pre-trained Gradient Boosting model & feature metadata
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation

