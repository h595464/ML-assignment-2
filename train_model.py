import json, joblib
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

data = fetch_ucirepo(id=73)
X = data.data.features.fillna("?")      # stalk-root has missing values
y = data.data.targets.iloc[:, 0]        # 'e' = edible, 'p' = poisonous

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("encode", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), list(X.columns))]
    )),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.joblib")
# save the valid options per feature so the form builds itself
options = {col: sorted(X[col].unique().tolist()) for col in X.columns}
json.dump(options, open("options.json", "w"), indent=2)