import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Last inn datasettet
data = pd.read_csv("train.csv")

# 2. Velg noen enkle features (variabler)
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
data = data[features + ["Survived"]]

# 3. Gjør om tekst til tall (Sex: male/female → 0/1)
data = pd.get_dummies(data, columns=["Sex"], drop_first=True)

# 4. Håndter manglende verdier
data = data.fillna(data.median())

# 5. Del opp i X (input) og y (mål)
X = data.drop("Survived", axis=1)
y = data["Survived"]

# 6. Tren / test-splitt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Tren modellen
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Test modellen
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", round(acc, 3))
