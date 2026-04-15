import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = int(request.form["smoker"])
    region = int(request.form["region"])

    # Prepare input for model
    input_data = np.array([[age, sex, bmi, children, smoker, 0, 0, 0]])

    # Adjust region encoding
    if region == 1:
        input_data[0][5] = 1
    elif region == 2:
        input_data[0][6] = 1
    elif region == 3:
        input_data[0][7] = 1

    # Predict insurance cost
    prediction = model.predict(input_data)


    # ---------------- Feature Importance Graph ----------------
    features = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]

    importances = model.feature_importances_

    plt.figure(figsize=(6,4))
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png")
    plt.close()


    # ---------------- Model Comparison Graph ----------------
    models = ["Linear Regression", "Random Forest"]
    scores = [0.78, 0.86]

    plt.figure(figsize=(5,4))
    plt.bar(models, scores)
    plt.title("Model Performance Comparison")
    plt.ylabel("R² Score")
    plt.savefig("static/model_comparison.png")
    plt.close()


    # ---------------- Age vs Charges Graph ----------------
    data = pd.read_csv("insurance.csv")

    plt.figure(figsize=(6,4))
    plt.scatter(data["age"], data["charges"], alpha=0.5)
    plt.title("Age vs Insurance Charges")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.savefig("static/age_vs_charges.png")
    plt.close()


    # Send result back to HTML
    return render_template(
        "index.html",
        prediction_text=f"Predicted Insurance Cost: ${round(prediction[0],2)}"
    )


if __name__ == "__main__":
    app.run(debug=True)
