# INT395 — Course Notes

Official course material for **Supervised Learning (INT395)**: lecture slides and the practical code that goes with them.

## Contents

- [`Slides/`](Slides/) — unit-wise lecture PDFs (`U1.pdf` through `U6.pdf`), plus `lecture-zero-INT395.pdf` (course intro) and a dedicated decision-tree slide deck.
- [`Code/`](Code/) — one notebook per practical (`P2` through `P13`), covering preprocessing, PCA, cross-validation, the core classification/regression algorithms, evaluation metrics, boosting, tuning, and ARIMA/SARIMA time series.

## Suggested order

Read the slide for a unit first, then open the matching notebook in `Code/` to see it implemented:

| Unit | Slide | Related notebook(s) |
|------|-------|----------------------|
| 0 | `lecture-zero-INT395.pdf` | — |
| 1 | `U1.pdf` | `EDA-pandas.ipynb`, `EDA-seaborn.ipynb` |
| 2 | `U2-INT395.pdf`, `U2-decision-tree.pdf` | `P2-Preprocessing.ipynb`, `P6-decision-tree.ipynb` |
| 3 | `U3.pdf` | `P3-PCA.ipynb`, `P4-cross-val.ipynb`, `P5-logistic.ipynb` |
| — | (regression) | `linear-reg.ipynb`, `polynomial-reg.ipynb`, `regularized-reg.ipynb`, `pipeline.ipynb` |
| 4 | `U4.pdf` | `P7-SVM-KNN-NB.ipynb`, `P7-comp.ipynb`, `P8-eval-met.ipynb` |
| 5 | `U5.pdf` | `P9-RF.ipynb`, `P10-boosting.ipynb`, `P11-tuning.ipynb` |
| 6 | `U6.pdf` | `P13-ARIMA-SARIMA.ipynb` |

For applied, hands-on repetitions of these same topics, see [`03-Practice/`](../03-Practice/) and [`02-Lessons/`](../02-Lessons/).
