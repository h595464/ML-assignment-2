import json, joblib
import pandas as pd
from flask import Flask, render_template, request
from labels import LABELS
app = Flask(__name__)
model = joblib.load("model.joblib")
options = json.load(open("options.json"))
display_options = {
    feature: [(c, LABELS.get(feature, {}).get(c, c)) for c in codes]
    for feature, codes in options.items()
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        row = {col: request.form[col] for col in options}
        df = pd.DataFrame([row])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        confidence = max(proba) * 100
        result = {
            "label": "Poisonous" if pred == "p" else "Edible (per model)",
            "poisonous": pred == "p",
            "confidence": round(confidence, 1),
        }
    return render_template("index.html", options=display_options, result=result)

if __name__ == "__main__":
    app.run(debug=True)