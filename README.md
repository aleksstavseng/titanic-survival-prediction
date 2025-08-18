# Titanic Survival Prediction

This project uses machine learning to predict which passengers survived the Titanic disaster, based on demographic and ticket-related information. It demonstrates skills in data processing, feature engineering, modeling, and evaluation.

---

## 🚀 Quick Results

| Model               | Accuracy (Hold-Out) | 5-Fold Cross Validation |
|--------------------|---------------------|--------------------------|
| LogisticRegression | 0.81                | –                        |
| RandomForest       | 0.83–0.85 (typical) | 0.82 ± 0.02              |

✅ **Best model:** RandomForest (~85% accuracy)

---

## 🎯 Goals

- Build and evaluate a model to predict survival.
- Apply advanced feature engineering to boost accuracy.
- Demonstrate a clear, professional ML pipeline.
- Show structured data insights with real-world application.

---

## 🛠️ Tech Stack

- **Programming:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **ML Models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Environment:** Jupyter Notebook / GitHub Codespaces

---

## 📂 Project Structure

```plaintext
.
├── data/                     # Raw dataset (train.csv, etc.)
├── titanic_model.py         # Main ML script
├── titanic.ipynb            # Jupyter Notebook (exploration & EDA)
├── requirements.txt         # Dependencies
├── README.md                # This file
└── .gitignore               # Files to ignore in Git
