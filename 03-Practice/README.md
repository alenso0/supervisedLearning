# Practice — Hands-on Notebooks by Topic

This folder used to be organized by class session date (`nov7`, `dec19k`, ...), which made it hard for anyone but the original author to find anything. It's now organized **by topic**, in roughly the order you'd learn them. Each numbered folder is self-contained — notebook(s) plus whatever CSV/data file they read sit side by side, so you can open a folder and run its notebook without chasing files elsewhere.

| # | Topic | Folder |
|---|-------|--------|
| 01 | Pandas & data handling basics | [`01-Pandas-and-Data-Handling/`](01-Pandas-and-Data-Handling/) |
| 02 | Data visualization | [`02-Data-Visualization/`](02-Data-Visualization/) |
| 03 | Data preprocessing (scaling, outliers, encoding, winsorization) | [`03-Data-Preprocessing/`](03-Data-Preprocessing/) |
| 04 | Feature engineering | [`04-Feature-Engineering/`](04-Feature-Engineering/) |
| 05 | Regression (linear, polynomial, ridge, grid search) | [`05-Regression/`](05-Regression/) |
| 06 | Classification (general, Naive Bayes, Perceptron) | [`06-Classification/`](06-Classification/) |
| 07 | Model evaluation & tuning (ROC, ROC-AUC) | [`07-Model-Evaluation-and-Tuning/`](07-Model-Evaluation-and-Tuning/) |
| 08 | Ensemble learning (voting, stacking) | [`08-Ensemble-Learning/`](08-Ensemble-Learning/) |
| 09 | Time series forecasting | [`09-Time-Series/`](09-Time-Series/) |
| 10 | POC algorithms (clean, tuned reference implementations) | [`10-POC-Algorithms/`](10-POC-Algorithms/) |

## Notes on this reorganization

- Several sub-sessions cover the same topic from different class dates (e.g. `05-Regression` has four sub-folders, `08-Ensemble-Learning` has three). These are kept as separate numbered sub-folders rather than merged, since merging risked silently overwriting one session's edits with another's.
- A handful of data files (`Hitters.csv`, `tvmarketing.csv`, the time-series CSVs) had been saved with Google-Drive-style `(1)`/`(7)` suffixes that no longer matched what the notebook code expected — those notebooks couldn't actually find their data. Files were renamed to match the code, and one notebook's data path was corrected, so everything in this folder should now run as-is.
- One genuinely empty/junk notebook (`Voting - Dec3/voting_dec3.ipynb`, 0 bytes) was removed; the real version of that work lives in `08-Ensemble-Learning/02-Voting-Extended-Discussion/`.
- Each topic folder still has its own datasets even where another folder happens to use the same dataset — that's intentional, so any single folder can be copied or run independently.

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels jupyter
jupyter notebook
```

Then open any notebook under the topic folder you're studying.
