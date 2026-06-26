## Supervised Learning (INT395) — Hands-on Repository

Welcome! This repository is a hands-on resource for **Supervised Learning**, originally built while working through the INT395 course. It contains course slides, a structured lesson curriculum, dozens of practice notebooks, and a small end-to-end capstone project.

It's organized so that **someone new to ML can start at the top and work down**, without needing any prior context about how the material was originally collected.

---

## 📚 Table of Contents

* [Repository Map](#-repository-map)
* [Suggested Learning Path](#-suggested-learning-path)
* [Setup and Usage](#-setup-and-usage)
* [Textbooks & References](#-textbooks--references)

---

## 🗺️ Repository Map

| Folder | What's in it |
|--------|--------------|
| [`01-Course-Notes-INT395/`](01-Course-Notes-INT395/) | Official INT395 lecture slides + matching practical notebooks |
| [`02-Lessons/`](02-Lessons/) | A full, ordered data-science → ML curriculum (13 lessons, slides + notebooks) |
| [`03-Practice/`](03-Practice/) | ~20 practice notebooks, grouped by topic (regression, classification, ensembles, time series, etc.) |
| [`04-Capstone-Project/`](04-Capstone-Project/) | "Hospital Bed Optimizer" — a full ML pipeline + Streamlit app + written research report |

Each folder above has its own `README.md` with more detail and a table of contents.

---

## 🧭 Suggested Learning Path

If you're learning this material for the first time, go in this order:

1. **Foundations** — [`02-Lessons/Lesson_01`](02-Lessons/Lesson_01_Introduction_to_Data_Science/) through [`Lesson_07`](02-Lessons/Lesson_07_Feature_Engineering/): data science basics, Pandas, visualization, math/stats, probability, data wrangling, and feature engineering.
2. **Core ML** — [`02-Lessons/Lesson_08`](02-Lessons/Lesson_08_Introduction_to_Machine_Learning/) through [`Lesson_13`](02-Lessons/Lesson_13_Model_deployment/): supervised learning (regression & classification), ensembles, time series, and deployment.
3. **Practice** — for each topic you just learned, find the matching folder in [`03-Practice/`](03-Practice/) and actually run the notebook yourself, end to end.
4. **Reference implementations** — [`03-Practice/10-POC-Algorithms/`](03-Practice/10-POC-Algorithms/) has clean, tuned, single-purpose notebooks for Naive Bayes, Decision Trees, SVM, Logistic Regression, and KNN — good for revision.
5. **Put it together** — [`04-Capstone-Project/`](04-Capstone-Project/) shows everything combined into a real pipeline with a UI, written up as a research report.

This covers, in practical terms:

* **Data Preprocessing & Visualization** 📊 — Pandas, Seaborn, handling missing values/outliers/scaling/encoding.
* **Feature Engineering & Selection** ⚙️ — including dimensionality reduction with PCA.
* **Model Implementation** 💻 — Logistic Regression, Decision Trees, k-NN, SVM, Naive Bayes, Ensembles (Random Forest, AdaBoost, Gradient Boosting, Voting, Stacking).
* **Model Evaluation & Tuning** 📈 — Precision/Recall/F1/ROC-AUC, Grid Search, Random Search.
* **Regression** 📉 — Linear, Polynomial, Ridge, and ARIMA/SARIMA time series forecasting.
* **Deployment** 🚀 — packaging a model with Pickle/joblib and serving it with Streamlit.

---

## 🛠️ Setup and Usage

### Prerequisites

* Python 3.x
* `pip`

### Installation

```bash
git clone https://github.com/alensomaxx/supervisedLearning.git
cd supervisedLearning

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install pandas numpy scikit-learn matplotlib seaborn statsmodels streamlit jupyter
jupyter notebook
```

Then open any notebook under `02-Lessons/`, `03-Practice/`, or `01-Course-Notes-INT395/Code/` — each one only depends on files in its own folder, so you can `cd` into any single topic folder and run it standalone.

To run the capstone project's interactive dashboard:

```bash
cd 04-Capstone-Project
pip install -r requirements.txt
streamlit run app.py
```

---

## 📖 Textbooks & References

1. **MACHINE LEARNING-I** by **CHANDRA S.S, VINOD**, PHI Learning (primary textbook)
2. **MACHINE LEARNING** by **ETHEM ALPAYDIN**, MIT Press (supplementary)

Datasets, lecture notes, and other supplementary files: [Google Drive](https://drive.google.com/drive/folders/1BAqJLdMl_oR1eNUwX1EiV_7qYiT7hXcs?usp=drive_link)

---

Happy Learning! 🚀
