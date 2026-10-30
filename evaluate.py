"""
Model comparison, baselines, metrics, error analysis and feature importance
for the mushroom classifier. Run: python evaluate.py

Positive class = "p" (poisonous). Recall on "p" is the safety-critical metric:
a missed poisonous mushroom (false negative) is far worse than a false alarm.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, make_scorer, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------- data
data = fetch_ucirepo(id=73)
X = data.data.features.fillna("?")
y = data.data.targets.iloc[:, 0]
cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ---------------------------------------------- non-ML baseline (a rule)
class OdorRule(BaseEstimator, ClassifierMixin):
    """Hand-written rule, no learning: edible if odor is almond, anise or none."""
    def fit(self, X, y):
        self.classes_ = np.array(["e", "p"])
        return self

    def predict(self, X):
        return np.where(X["odor"].isin(["a", "l", "n"]), "e", "p")


def pipe(clf, columns=cols):
    return Pipeline([
        ("encode", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), columns)])),
        ("clf", clf),
    ])


models = {
    "Majority class (baseline)": DummyClassifier(strategy="most_frequent"),
    "Odor rule (non-ML baseline)": OdorRule(),
    "Decision tree, depth 3": pipe(DecisionTreeClassifier(max_depth=3, random_state=42)),
    "Logistic regression": pipe(LogisticRegression(max_iter=1000)),
    "Decision tree, full depth": pipe(DecisionTreeClassifier(random_state=42)),
    "Random forest": pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
}

# ------------------------------------------------ metrics: holdout + CV
POS = "p"
scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=POS, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=POS, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=POS, zero_division=0),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows, fitted, preds = [], {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted[name], preds[name] = model, pred

    cvres = cross_validate(model, X, y, cv=cv, scoring=scoring)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=["e", POS]).ravel()
    rows.append({
        "model": name,
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, pos_label=POS, zero_division=0),
        "test_recall": recall_score(y_test, pred, pos_label=POS, zero_division=0),
        "test_f1": f1_score(y_test, pred, pos_label=POS, zero_division=0),
        "missed_poisonous (FN)": fn,
        "false_alarms (FP)": fp,
        "cv_accuracy_mean": cvres["test_accuracy"].mean(),
        "cv_accuracy_std": cvres["test_accuracy"].std(),
        "cv_recall_mean": cvres["test_recall"].mean(),
    })

results = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print("\n=== Results (positive class = poisonous) ===")
print(results.round(4).to_string())
results.round(4).to_csv("results.csv")

# ------------------------------------------------- confusion matrices
shown = ["Odor rule (non-ML baseline)", "Decision tree, depth 3", "Random forest"]
fig, axes = plt.subplots(1, len(shown), figsize=(13, 4))
for ax, name in zip(axes, shown):
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[name], labels=["e", POS],
        display_labels=["edible", "poisonous"], ax=ax, colorbar=False)
    ax.set_title(name, fontsize=10)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# ----------------------------------------------------- error analysis
errors = []
for name, pred in preds.items():
    wrong = X_test[pred != y_test].copy()
    wrong["true"] = y_test[pred != y_test]
    wrong["predicted"] = pred[pred != y_test]
    wrong["model"] = name
    errors.append(wrong)
errors = pd.concat(errors)
errors.to_csv("errors.csv")
print("\n=== Misclassified test samples per model ===")
print(errors.groupby("model").size().to_string())

fn_rows = errors[(errors["true"] == POS) & (errors["predicted"] == "e")]
if len(fn_rows):
    print("\nPoisonous mushrooms predicted as edible (most dangerous errors):")
    print(fn_rows.groupby("model")["odor"].value_counts().to_string())
else:
    print("\nNo poisonous mushroom was predicted as edible by any model on the test set.")

# --------------------------------------------------- feature importance
rf = fitted["Random forest"]
enc, clf = rf.named_steps["encode"], rf.named_steps["clf"]
names = enc.get_feature_names_out()                       # e.g. cat__odor_n
group = [n.split("__", 1)[1].rsplit("_", 1)[0] for n in names]
impurity = (pd.Series(clf.feature_importances_, index=group)
            .groupby(level=0).sum().sort_values(ascending=False))

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
permutation = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)

print("\n=== Top 10 features (impurity importance, summed per original feature) ===")
print(impurity.head(10).round(4).to_string())
print("\n=== Top 10 features (permutation importance, accuracy drop) ===")
print(permutation.head(10).round(4).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
impurity.head(10)[::-1].plot.barh(ax=axes[0], title="Impurity importance")
permutation.head(10)[::-1].plot.barh(ax=axes[1], title="Permutation importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ------------------- using importance: do we need all 22 features?
print("\n=== Random forest using only the top-k features (holdout) ===")
subset_rows = []
for k in [1, 2, 3, 5, 8, len(cols)]:
    top = list(impurity.index[:k])
    m = pipe(RandomForestClassifier(n_estimators=100, random_state=42), top)
    m.fit(X_train[top], y_train)
    p = m.predict(X_test[top])
    subset_rows.append({
        "k": k,
        "accuracy": accuracy_score(y_test, p),
        "recall_poisonous": recall_score(y_test, p, pos_label=POS),
        "features": ", ".join(top) if k <= 5 else "...",
    })
subset = pd.DataFrame(subset_rows)
print(subset.round(4).to_string(index=False))
subset.round(4).to_csv("feature_subset_results.csv", index=False)

print("\nSaved: results.csv, errors.csv, feature_subset_results.csv, "
      "confusion_matrices.png, feature_importance.png")
