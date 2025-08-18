# Titanic Survival Prediction 🚢

## 🚀 Hurtigresultater
| Modell              | Accuracy (hold-out) | 5-fold CV   |
|---------------------|---------------------|-------------|
| LogisticRegression  | 0.81                | –           |
| RandomForest        | 0.83–0.85 (typisk)  | 0.82 ± 0.02 |

**Beste modell:** RandomForest (~85% accuracy)

---

## 📝 Oversikt
Dette prosjektet demonstrerer en komplett **maskinlærings-pipeline** i Python, med Titanic-datasettet.  
Målet er å predikere om en passasjer overlevde Titanic-katastrofen basert på demografi og billettinformasjon.

---

## 🛠️ Tech Stack
- **Programmering:** Python  
- **Biblioteker:** Pandas, NumPy, scikit-learn, Matplotlib, Seaborn  
- **Miljø:** Jupyter Notebook / Google Colab / Codespaces  

---

## ▶️ Hvordan kjøre
```bash
pip install -r requirements.txt
python titanic_model.py
