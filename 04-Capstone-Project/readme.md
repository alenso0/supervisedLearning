# 🏥 Hospital Bed Optimizer

**An ML-powered system to predict patient length of stay and optimize hospital resource allocation**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-Educational-green)

## 📋 Table of Contents
- [Executive Summary](#executive-summary)
- [Data Provenance](#data-provenance--citation)
- [Methodology](#methodology--pipeline)
- [Performance Evaluation](#performance-evaluation)
- [Technical Architecture](#technical-architecture)
- [Installation & Usage](#installation--usage)
- [Future Scope](#future-scope)
- [License](#license)

## 🚀 Quick Start
```bash
git clone https://github.com/thesanyamjain007/hospital-bed-optimization.git
cd hospital-bed-optimization
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Key Results
- **Best Model**: SVM with 81.82% ROC-AUC
- **Decision Tree** offers optimal recall (71%) for identifying long stays
- **Test Accuracy Range**: 47.67% - 78.67% across all algorithms

## 🔧 Tech Stack
- **ML Framework**: scikit-learn
- **Frontend**: Streamlit
- **Language**: Python
- **Serialization**: joblib

## 📁 Project Structure
```
hospital-bed-optimization/
├── app.py                 # Streamlit dashboard
├── backend_models.py      # ML pipeline & model training
├── requirements.txt       # Dependencies
└── README.md
```

## 💡 Key Features
✅ Multi-algorithm benchmarking (5 classifiers)  
✅ Interactive correlation heatmap  
✅ Automated data preprocessing  
✅ CSV batch prediction export  
✅ Real-time model evaluation metrics  

## 🤝 Contributing
Contributions welcome! Please submit issues or pull requests to improve model performance or add new features.

---
## 1. Executive Summary

Effective hospital resource management relies on anticipating patient flow. This project implements a machine learning pipeline to classify hospitalizations as **Short Stay (0-7 days)** or **Long Stay (>7 days)**, optimizing bed allocation, staffing, and inventory management.

Features an interactive Streamlit dashboard for data ingestion, preprocessing, EDA, multi-algorithm training, and inference.

## 2. Data Provenance & Citation

**Original Source**: Microsoft R Server - Hospital Length of Stay  
**Dataset Repository**: [Kaggle - Hospital Length of Stay Dataset (Microsoft)](https://www.kaggle.com/datasets/microsoft/hospital-length-of-stay)

**Citation**:
```
Microsoft Corporation. (2016). Hospital Length of Stay Data. 
Retrieved from Microsoft/Kaggle.
```

De-identified patient encounter data including demographics, clinical history, and administrative details.

## 3. Methodology & Pipeline

### 3.1 Data Preprocessing & Feature Engineering

**Dimensionality Reduction**: Identified and dropped irrelevant features to reduce noise.

**Dropped Columns**:
- `eid` (Encounter ID)
- `vdate` (Visit Date)
- `discharged` (Discharge Date)
- `facid` (Facility ID)

**Categorical Encoding**:
- `gender`: Binary encoded ($F \rightarrow 0, M \rightarrow 1$)
- `rcount` (Readmission Count): String to integer conversion

**Target Engineering**:
- $0$: Length of Stay $\le 7$ days
- $1$: Length of Stay $> 7$ days

### 3.2 Exploratory Data Analysis

Interactive Correlation Heatmap identifies multicollinearity and feature importance. Strong correlations observed between `long_stay_label` and `rcount`, `psychologicaldisordermajor`.

### 3.3 Algorithm Selection

Five classifiers benchmarked:
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

## 4. Performance Evaluation

| Algorithm | Train Accuracy | Test Accuracy | ROC AUC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-----------|----------------|---------------|---------|---------------------|------------------|-------------------|
| SVM | 0.8371 | 0.7517 | 0.8182 | 0.8405 | 0.6667 | 0.7435 |
| Decision Tree | 0.8325 | 0.7867 | 0.8100 | 0.8792 | 0.7099 | 0.8014 |
| Logistic Regression | 0.8279 | 0.7350 | 0.8008 | 0.7855 | 0.7006 | 0.7406 |
| Naive Bayes | 0.7900 | 0.7083 | 0.6972 | 0.7899 | 0.6265 | 0.6988 |
| KNN | 0.6608 | 0.4767 | 0.5074 | 0.5217 | 0.3704 | 0.4332 |

**Key Findings**:
- **Top Performer**: SVM (83.7% Train Accuracy, 0.81 ROC AUC)
- **Best Recall**: Decision Tree (71%) for identifying long stays
- **Limitation**: KNN underperformed due to dataset dimensionality

## 5. Technical Architecture

**Modular Python Architecture**:
- `app.py`: Frontend interface (Streamlit)
- `backend_models.py`: ML pipeline, sklearn integration, model serialization

## 6. Operational Flow

1. Upload `LengthOfStay.csv`
2. Review Correlation Matrix
3. Click "Train & Evaluate Models"
4. Download `predictions.joblib`

## 7. Future Scope

- Hyperparameter tuning (GridSearch, Bayesian Optimization)
- Ensemble methods (Random Forests, XGBoost)
- Deep learning for non-linear interactions
- Advanced feature engineering

## 8. License

Research and educational purposes. Dataset governed by Microsoft Research License.

**Authors**: Alen Alex, Sanyam Jain, Mansi Pandey  