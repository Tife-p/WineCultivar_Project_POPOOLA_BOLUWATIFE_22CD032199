from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["alcohol"]),
                float(request.form["malic_acid"]),
                float(request.form["ash"]),
                float(request.form["magnesium"]),
                float(request.form["flavanoids"]),
                float(request.form["color_intensity"])
            ]

            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            result = model.predict(features_scaled)[0]

            prediction = f"Cultivar {result + 1}"

        except:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
