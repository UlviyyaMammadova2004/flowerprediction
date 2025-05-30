from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        try:
            # Formdan verileri alir
            f1 = request.form["f1"]
            f2 = request.form["f2"]
            f3 = request.form["f3"]
            f4 = request.form["f4"]

            # Float'a çevirip tahmin yapar
            data = np.array([[float(f1), float(f2), float(f3), float(f4)]])
            prediction = model.predict(data)[0]

            # Sonuçla birlikte inputları tekrar form sayfasına gönderir
            return render_template(
                "form.html",
                prediction=prediction,
                f1=f1, f2=f2, f3=f3, f4=f4
            )
        except Exception as e:
            return render_template("form.html", error=str(e))
    
    # GET methodu için boş form yaratir
    return render_template("form.html")
