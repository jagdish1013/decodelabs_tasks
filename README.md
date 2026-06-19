# 💳 Credit Card Fraud Detection using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?style=for-the-badge&logo=numpy)
![Groq](https://img.shields.io/badge/Groq-LLM%20Integration-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Project Overview

Financial fraud causes billions of dollars in losses every year. Due to the highly imbalanced nature of transaction datasets, detecting fraudulent transactions accurately remains a major challenge.

This project presents an end-to-end Machine Learning pipeline for detecting fraudulent credit card transactions using advanced preprocessing techniques, SMOTE-based class balancing, hyperparameter optimization, and comparative evaluation of multiple models.

The objective is to maximize fraud detection capability while minimizing false alarms.

---

# 🎯 Objectives

- Detect fraudulent credit card transactions.
- Handle severe class imbalance effectively.
- Compare multiple machine learning models.
- Optimize model performance using GridSearchCV.
- Improve recall without sacrificing precision significantly.
- Build a reusable fraud detection pipeline.

---

# 📂 Dataset Information

Dataset contains anonymized credit card transactions.

### Features

- Total Features: **31**
- Numerical Features: **30**
- Target Variable:

| Variable | Description |
|-----------|-------------|
| Class | 0 = Legitimate Transaction |
| Class | 1 = Fraudulent Transaction |

### Dataset Shape

```
(53,570 Rows × 31 Columns)
```

### Class Distribution

| Class | Count | Percentage |
|---------|---------|------------|
| Legitimate | 53,417 | 99.71% |
| Fraud | 153 | 0.29% |

This extreme imbalance makes fraud detection particularly challenging.

---

# 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

## ✔ Dataset Inspection

- Data shape analysis
- Feature type inspection
- Statistical summary
- Missing value detection

## ✔ Fraud Distribution Analysis

Visualized transaction imbalance using:

- Count plots
- Percentage distribution

## ✔ Correlation Analysis

Generated correlation heatmaps to understand relationships among variables.

---

# ⚙️ Data Preprocessing Pipeline

## Missing Value Handling

Missing records were found in:

- V23
- V24
- V25
- V26
- V27
- V28
- Amount
- Class

Strategy used:

```python
df.dropna(subset=['Class'], inplace=True)
```

---

## Train-Test Split

```python
80% → Training
20% → Testing
```

Using:

```python
train_test_split(
    stratify=y,
    random_state=42
)
```

Stratification preserves fraud proportions.

---

# ⚖️ Handling Class Imbalance

The dataset is heavily skewed.

Before SMOTE:

```
Non-Fraud : 42,734
Fraud     : 122
```

SMOTE (Synthetic Minority Oversampling Technique) was applied to generate synthetic fraud samples.

### Benefits

✅ Better fraud representation

✅ Improved recall

✅ Reduced bias toward majority class

---

# 🤖 Machine Learning Models

## 1️⃣ Logistic Regression Pipeline

Pipeline Components:

```
StandardScaler
↓
SMOTE
↓
Logistic Regression
```

Hyperparameters tuned:

```python
C = [0.01, 0.1, 1, 10]

SMOTE k_neighbors = [3,5,7]
```

Best Parameters:

```python
{
 'classifier__C': 10,
 'smote__k_neighbors': 3
}
```

---

## Logistic Regression Results

| Metric | Score |
|----------|--------|
| Precision | 0.164 |
| Recall | 0.903 |
| ROC-AUC | 0.967 |

### Interpretation

- Excellent fraud capture ability.
- Higher false positives.
- Suitable when missing fraud is costly.

---

# 2️⃣ Random Forest Pipeline

Pipeline Components:

```
SMOTE
↓
Random Forest
```

Hyperparameters tuned:

```python
n_estimators = [100,200]

max_depth = [10,20,None]

SMOTE k_neighbors = [5]
```

Best Parameters:

```python
{
 'classifier__max_depth': 10,
 'classifier__n_estimators': 100,
 'smote__k_neighbors': 5
}
```

---

# 🌟 Random Forest Results

| Metric | Score |
|----------|--------|
| Precision | 0.659 |
| Recall | 0.935 |
| ROC-AUC | 0.997 |

### Interpretation

- Outstanding fraud detection capability.
- Significantly fewer false alarms.
- Best overall performer.

---

# 📊 Model Comparison

| Model | Precision | Recall | ROC-AUC |
|---------|------------|----------|-----------|
| Logistic Regression | 0.164 | 0.903 | 0.967 |
| Random Forest | ⭐ 0.659 | ⭐ 0.935 | ⭐ 0.997 |

---

# 🏆 Best Model

## Random Forest Classifier

Selected because it achieved:

- Highest ROC-AUC
- Highest Recall
- Better Precision
- Strong overall balance

---

# 📈 Evaluation Metrics Used

- Precision
- Recall
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- ROC Curve

---

# 🧠 Project Architecture

```
Credit Card Dataset
        │
        ▼
 Exploratory Data Analysis
        │
        ▼
 Missing Value Handling
        │
        ▼
 Train-Test Split
        │
        ▼
     SMOTE
        │
 ┌──────┴──────┐
 ▼             ▼
Logistic    Random
Regression   Forest
 │             │
 ▼             ▼
GridSearchCV Optimization
        │
        ▼
Performance Evaluation
        │
        ▼
Best Fraud Detection Model
```

---

# ⚡ Groq Integration

This project documentation and analytical explanations can be enhanced using Groq-powered Large Language Models.

### Groq Benefits

- Ultra-fast inference.
- Real-time model explanations.
- AI-assisted fraud insights.
- Automated reporting.
- Natural language interpretation of predictions.

Potential Use Cases:

- Explain why a transaction was flagged.
- Generate fraud investigation summaries.
- Convert technical outputs into business reports.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- GridSearchCV
- Groq API

---

# 🚀 Future Improvements

- Deploy using Streamlit or Flask.
- Integrate real-time transaction monitoring.
- Add XGBoost and LightGBM.
- Explain predictions using SHAP.
- Use Groq LLM for AI-powered fraud explanations.

---

# 📬 Contact

### Jagdish

AI/ML Engineer

📧 (jagdishmeghwal1013@gmail.com)


## ⭐ If you found this project useful, consider giving it a star!
