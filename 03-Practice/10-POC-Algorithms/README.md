# 📚 Supervised Learning Practice & Proof-of-Concepts (POCs)

This repository is a focused learning environment for mastering **core Supervised Machine Learning algorithms** through hands-on implementation, hyperparameter tuning, and performance evaluation — all using the Python data science stack (primarily **Scikit-learn** and **Jupyter Notebooks**).

The main goal is to provide **clean, well-documented, and fully reproducible** examples of foundational algorithms — perfect for learning, revision, or as reference implementations.

---

### 🎯 Proof-of-Concept (POC) Algorithms

All POCs are located in the [`Practice/POC/`](Practice/POC/) directory. Each algorithm has its own dedicated, fully commented Jupyter Notebook.

| Algorithm                  | Notebook                          | Status                  | Focus                                      |
|----------------------------|-----------------------------------|-------------------------|--------------------------------------------|
| Naive Bayes                | [`NaiveBayesAlgo.ipynb`](https://github.com/alensomaxx/supervisedLearning/blob/main/Practice/POC/NaiveBayesAlgo.ipynb)           | ✅ POC Complete        | Implementation + Initial Evaluation         |
| Decision Trees             | [`DecisionTreesAlgo.ipynb`](https://github.com/alensomaxx/supervisedLearning/blob/main/Practice/POC/DecisionTressAlgo.ipynb)       | ✅ Tuned               | Implementation + Hyperparameter Tuning     |
| Support Vector Machines    | [`SVMAlgo.ipynb`](https://github.com/alensomaxx/supervisedLearning/blob/main/Practice/POC/SVMAlgo.ipynb)                   | ✅ Tuned               | Implementation + Hyperparameter Tuning     |
| Logistic Regression        | [`LogisticRegressionAlgo.ipynb`](https://github.com/alensomaxx/supervisedLearning/blob/main/Practice/POC/LogisticRegressionAlgo.ipynb) | ✅ Tuned               | Implementation + Hyperparameter Tuning     |
| K-Nearest Neighbors (KNN)  | [`KNNAlgo.ipynb`](https://github.com/alensomaxx/supervisedLearning/blob/main/Practice/POC/KNNAlgo.ipynb)                 | ✅ Tuned               | Implementation + Hyperparameter Tuning     |

> All models are trained and evaluated on the **Wisconsin Breast Cancer Dataset** (`Breast_cancer_dataset.csv`) — a classic binary classification benchmark.


### 🚀 Getting Started

Ready to run the notebooks locally? Just follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/alensomaxx/supervisedLearning.git
cd supervisedLearning

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook
```
Then navigate to `Practice/POC/` and open any notebook (e.g., `KNNAlgo.ipynb`) to explore, run, and modify!

---

### 🤝 Contribution

This repository is primarily for **personal learning and practice**.  
External contributions via Pull Requests are not accepted, but **feedback, suggestions, or corrections** are very welcome!  
Feel-contrast free to open an [Issue](https://github.com/alensomaxx/supervisedLearning/issues) anytime.

---

### ⭐ Star this repo if you found it helpful for learning supervised ML algorithms!

**Happy learning!** 🚀
