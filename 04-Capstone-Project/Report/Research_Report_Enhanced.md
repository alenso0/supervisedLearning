# Hospital Bed Capacity Optimization: A Comparative Analysis of Machine Learning Classifiers for Length of Stay Prediction

**Date:** November 28, 2025  
**Project Status:** Phase 1 (Proof of Concept) - Completed  
**Dataset:** `dummy_hospital_beds_200.csv`  

---

## Abstract

Effective hospital resource management is paramount for maintaining patient care standards and operational efficiency. This study presents a Proof of Concept (POC) designed to optimize bed allocation by predicting patient Length of Stay (LOS). Utilizing a synthetic dataset of 200 patient records, we engineered a comprehensive data processing pipeline and evaluated the performance of five distinct machine learning algorithms. The experimental results demonstrate the efficacy of the proposed solution in classifying patients into "Long Stay" versus "Short Stay" categories. Notably, the study employs Receiver Operating Characteristic (ROC) analysis to determine optimal decision thresholds, ensuring a balance between sensitivity and specificity. This report details the methodology, experimental configuration, and statistical evaluation of the Phase 1 milestone.

---

## 1. Introduction

The efficient management of hospital bed capacity is a critical challenge in modern healthcare administration. Overcrowding and inefficient bed turnover can lead to increased wait times, admission bottlenecks, and compromised patient outcomes. The primary objective of this research is to develop a predictive model capable of identifying patients likely to require an extended hospital stay (`long_stay_label`) based on admission demographics and clinical indicators.

By accurately forecasting these instances, hospital administration can proactively manage staffing levels and bed availability. This document outlines the end-to-end solution architecture, ranging from data ingestion and preprocessing to the comparative statistical evaluation of five classification algorithms.

---

## 2. Methodology: Data Pipeline Architecture

The solution architecture is governed by a strict pipeline designed to ensure data integrity, reproducibility, and scalability. The system logic is compartmentalized into backend processing (`backend_models.py`) and frontend visualization (`app.py`).

### 2.1 Data Collection and Inspection
The study utilizes `dummy_hospital_beds_200.csv` as the primary data source. The system is engineered to support dynamic data loading via a user interface, compatible with both CSV and Excel formats. Initial inspection of the dataset identified key features including `age`, `gender`, `department`, `admission_type`, and `severity_score`, alongside the binary target variable `long_stay_label`.

### 2.2 Data Preprocessing and Cleaning
To prepare the raw data for algorithmic training, a rigorous cleaning protocol was implemented within the `process_dataset` function:

*   **Missing Value Imputation:** To address data sparsity without discarding valuable records, missing numeric values were imputed using the **median**, offering robustness against skew. Categorical missing values were filled using the **mode** (most frequent value).
*   **Outlier Management:** Numerical features were subjected to Interquartile Range (IQR) clipping. Values falling below 1 - 1.5 \times  or exceeding 1.5 \times were capped. This step is crucial for preventing extreme anomalies from distorting the decision boundaries of the models.
*   **Feature Encoding:**
    *   **Label Encoding:** Applied to binary categorical variables (e.g., Gender) to convert text labels into machine-readable numeric format.
    *   **One-Hot Encoding:** Applied to multi-class nominal variables (e.g., Department) to prevent the model from inferring an ordinal relationship where none exists.
*   **Dimensionality Reduction:** Non-predictive identifiers, such as `patient_id`, were excised to reduce noise and computational overhead.

### 2.3 Exploratory Data Analysis (EDA)
Prior to model training, an exploratory analysis was conducted to elucidate underlying data structures:
1.  **Correlation Heatmap:** Generated to detect multicollinearity among numerical features.
2.  **Target Distribution Countplot:** Used to assess class balance within the `long_stay_label`, ensuring the dataset was not heavily skewed toward one outcome.
3.  **Demographic Histograms:** Analyzed the distribution of age and other demographics across stay labels.

---

## 3. Experimental Setup

### 3.1 Stratified Train-Test Split
To evaluate model generalization capabilities, the dataset was partitioned using a **70:30 ratio**:
*   **Training Set:** 140 samples (70%) used for model fitting.
*   **Testing Set:** 60 samples (30%) reserved for validation.
*   **Stratification:** The split was stratified based on the `long_stay_label`. This ensures that the proportion of Long Stay vs. Short Stay patients remains consistent across both training and testing sets, preventing evaluation bias.

### 3.2 Feature Scaling
Feature scaling was applied using `StandardScaler` within a scikit-learn Pipeline. This process standardizes features by removing the mean and scaling to unit variance. This step is mathematically critical for distance-based algorithms (such as KNN and SVM), which are otherwise sensitive to the magnitude of raw feature values.

### 3.3 Algorithms Evaluated
Five distinct supervised learning algorithms were selected to provide a comprehensive comparative analysis:
1.  **Logistic Regression:** Utilized as a statistical baseline for binary classification.
2.  **Naive Bayes (GaussianNB):** A probabilistic classifier leveraging Bayes' theorem, assuming feature independence.
3.  **K-Nearest Neighbors (KNN):** A non-parametric, distance-based classifier (configured with k=5).
4.  **Decision Tree:** A non-parametric model that splits data based on feature values to maximize information gain.
5.  **Support Vector Machine (SVM):** A robust classifier utilizing the Radial Basis Function (RBF) kernel to handle non-linear decision boundaries.

---

## 4. Statistical Evaluation and Results

The performance of each algorithm was evaluated on the held-out Test Set (30%). Metrics recorded include Accuracy, Precision, Recall, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

### 4.1 Performance Metrics Overview
*   **Accuracy:** The ratio of correctly predicted observations to total observations.
*   **ROC-AUC:** A performance measurement for classification problems at various threshold settings. An AUC of 1.0 represents a perfect model, while 0.5 represents a random guess.
*   **Optimal Threshold:** Calculated as argmax(TPR - FPR), this value represents the probability threshold that maximizes the True Positive Rate while minimizing the False Positive Rate.

### 4.2 Comparative Results Table

The following table summarizes the performance metrics for all five algorithms.

| Algorithm | Train Accuracy | Test Accuracy | ROC AUC | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.986 | 0.983 | 1.000 | 0.972 | 1.000 | 0.986 |
| **Naive Bayes** | 0.971 | 0.983 | 0.999 | 0.972 | 1.000 | 0.986 |
| **Decision Tree** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **SVM** | 0.986 | 0.900 | 0.976 | 0.892 | 0.943 | 0.917 |
| **KNN** | 0.821 | 0.750 | 0.839 | 0.813 | 0.743 | 0.776 |

### 4.3 Analysis of Results
*   **Top Performers:** The **Decision Tree** model achieved perfect classification scores (1.0 across all metrics). While this indicates flawless performance on the current dataset, it may suggest overfitting or that the synthetic dataset contains highly distinct decision boundaries. **Logistic Regression** and **Naive Bayes** also exhibited exceptional performance with Test Accuracies of 98.3% and near-perfect AUC scores.
*   **Support Vector Machine (SVM):** The SVM model performed robustly with a 90% Test Accuracy and an AUC of 0.976, indicating strong separability capabilities, though slightly less precise than the linear models in this specific context.
*   **K-Nearest Neighbors (KNN):** KNN was the least effective model, with a Test Accuracy of 75% and an AUC of 0.839. This suggests that the spatial distribution of the data points may not be ideal for distance-based clustering without further feature engineering or dimensionality reduction.

---

## 5. Conclusion and Phase 1 Status

**Phase 1 Status:** **SUCCESS**

The Proof of Concept successfully validated the feasibility of using machine learning for hospital bed capacity optimization. The automated pipeline demonstrated the ability to ingest, clean, and process patient data efficiently. The comparative analysis identified the Decision Tree and Logistic Regression models as the most promising candidates for deployment, given their high accuracy and AUC scores.

**Deliverables Completed:**
*   [x] Data Ingestion & Cleaning Pipeline.
*   [x] Exploratory Data Analysis Dashboard.
*   [x] Multi-Algorithm Training and Evaluation.
*   [x] Statistical Evaluation (ROC, AUC, Confusion Matrix).
*   [x] Prediction Interface for new data.

The system is fully operational for the scope of Phase 1 and is ready for Phase 2, which will involve integration with live hospital databases and validation against larger, real-world datasets.

### 6. Software Availability
To ensure reproducibility, the project code is organized as follows:
*   **`backend_models.py`:** Contains core logic for data loading, imputation, outlier removal, encoding, stratified splitting, and model training.
*   **`app.py`:** Provides the user interface for EDA visualization, model training triggers, and prediction display.
*   **`dummy_hospital_beds_200.csv`:** The synthetic dataset utilized for this POC validation.