 from pathlib import Path
import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Stier relativt til denne fila (funker uansett hvor du kjører fra)
BASE = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "data" / "train.csv"
TEST_PATH  = BASE / "data" / "test.csv"
assert TRAIN_PATH.exists(), "Mangler data/train.csv"
assert TEST_PATH.exists(),  "Mangler data/test.csv"

# Les inn
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y = train["Survived"].astype(int)
train_X = train.drop(columns=["Survived"])
test_X  = test.copy()

# Samle for konsistent feature engineering
full = pd.concat([train_X, test_X], axis=0, ignore_index=True)

# --- Feature engineering (kun fra felter i datasettet) ---
def extract_title(name: str) -> str:
    m = re.search(r",\s*([^\.]+)\.", str(name))
    return m.group(1).strip() if m else "Unknown"

full["Title"] = full["Name"].apply(extract_title)
title_map = {
    "Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare","Sir":"Rare",
    "Jonkheer":"Rare","Don":"Rare","Dona":"Rare","Capt":"Officer","Col":"Officer",
    "Major":"Officer","Dr":"Officer","Rev":"Officer"
}
full["Title"] = full["Title"].replace(title_map)
common = {"Mr","Mrs","Miss","Master","Officer","Rare"}
full["Title"] = full["Title"].where(full["Title"].isin(common), "Rare")

full["FamilySize"] = full["SibSp"].fillna(0) + full["Parch"].fillna(0) + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

full["Ticket_str"] = full["Ticket"].astype(str)
full["TicketPrefix"] = full["Ticket_str"].str.replace(r"[0-9\.\/ ]", "", regex=True).str.upper().replace("", "NONE")
ticket_counts = full["Ticket_str"].value_counts()
full["TicketGroupSize"] = full["Ticket_str"].map(ticket_counts).clip(lower=1, upper=8)

full["CabinDeck"] = full["Cabin"].astype(str).str[0].str.upper()
full.loc[full["Cabin"].isna(), "CabinDeck"] = "Unknown"
full["HasCabin"] = full["Cabin"].notna().astype(int)

full["Fare"] = full["Fare"].astype(float).fillna(full["Fare"].median())
full["FarePerPerson"] = full["Fare"] / full["FamilySize"].replace(0, 1)
full["FareLog"] = np.log1p(full["Fare"])
full["FarePerPersonLog"] = np.log1p(full["FarePerPerson"])

def impute_age_by_title(df: pd.DataFrame) -> pd.Series:
    out = df["Age"].copy()
    grp_med = df.groupby("Title")["Age"].median()
    return out.fillna(df["Title"].map(grp_med).fillna(df["Age"].median()))
full["Age"] = impute_age_by_title(full)

full["Pclass_Sex"] = full["Pclass"].astype(str) + "_" + full["Sex"].astype(str)

cat_cols = ["Sex","Embarked","Title","CabinDeck","TicketPrefix","Pclass_Sex"]
num_cols = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone",
            "TicketGroupSize","FarePerPerson","FareLog","FarePerPersonLog"]
full_model = full[cat_cols + num_cols].copy()

train_features = full_model.iloc[:len(train_X)].reset_index(drop=True)
test_features  = full_model.iloc[len(train_X):].reset_index(drop=True)

# --- Preprosess + modeller ---
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)])

hgb = HistGradientBoostingClassifier(random_state=42, max_depth=3, max_iter=400, learning_rate=0.06)
xgb = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
lr = LogisticRegression(max_iter=1000)

ensemble = VotingClassifier(estimators=[("hgb", hgb), ("xgb", xgb), ("lr", lr)],
                            voting="soft", weights=[2,2,1])
pipe = Pipeline([("pre", pre), ("clf", ensemble)])

# --- 5-fold CV ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (tr, va) in enumerate(cv.split(train_features, y), 1):
    X_tr, X_va = train_features.iloc[tr], train_features.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, pred)
    scores.append(acc)
    print(f"Fold {i} accuracy: {acc:.3f}")
print(f"Mean CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# --- Hold-out ---
X_tr, X_te, y_tr, y_te = train_test_split(train_features, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
hold_acc = accuracy_score(y_te, pipe.predict(X_te))
print(f"Hold-out accuracy: {hold_acc:.3f}")

# --- Tren på alt + pseudo-labeling ---
pipe.fit(train_features, y)
test_proba = pipe.predict_proba(test_features)[:,1]
idx, labels = [], []
for i, p in enumerate(test_proba):
    if p >= 0.98: idx.append(i); labels.append(1)
    elif p <= 0.02: idx.append(i); labels.append(0)
print(f"Pseudo-labeling på {len(idx)} test-rader")
if idx:
    X_pl = pd.concat([train_features, test_features.iloc[idx]], axis=0).reset_index(drop=True)
    y_pl = pd.concat([y, pd.Series(labels, index=np.arange(len(idx)))], axis=0).reset_index(drop=True)
    pipe.fit(X_pl, y_pl)

final_pred = pipe.predict(test_features)

# --- Lagre ut ---
(sub_path := BASE / "submission.csv").write_text(
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_pred}).to_csv(index=False)
)
os.makedirs(BASE / "results", exist_ok=True)
os.makedirs(BASE / "models", exist_ok=True)
with open(BASE / "results" / "model_report.md", "w") as f:
    f.write("# Model Report\n")
    f.write(f"- Mean CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}\n")
    f.write(f"- Hold-out accuracy: {hold_acc:.3f}\n")
    f.write(f"- Pseudo-labeled rows: {len(idx)}\n")
joblib.dump(pipe, BASE / "models" / "ensemble_pipeline.pkl")
print("Skrev submission.csv, results/model_report.md og models/ensemble_pipeline.pkl")
