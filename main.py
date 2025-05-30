from flask import Flask, request, render_template
import numpy as np
import joblib
import os
from pymongo import MongoClient

app = Flask(__name__)
model = joblib.load("iris_model.pkl")

# MongoDB bağlantısı
mongo_uri = "mongodb+srv://mammadovaulia82:Jungkook_19971997@cluster0.2g47tbp.mongodb.net/tasklist"
client = MongoClient(mongo_uri)
db = client["flowerdb"]
collection = db["predictions"]

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        try:
            # Kullanıcıdan gelen verileri al
            f1 = request.form["f1"]
            f2 = request.form["f2"]
            f3 = request.form["f3"]
            f4 = request.form["f4"]

            # Float'a çevir ve modelle tahmin yap
            data = np.array([[float(f1), float(f2), float(f3), float(f4)]])
            prediction = model.predict(data)[0]

            # MongoDB'ye kaydet
            result = {
                "sepal_length": float(f1),
                "sepal_width": float(f2),
                "petal_length": float(f3),
                "petal_width": float(f4),
                "predicted_class": int(prediction)
            }
            collection.insert_one(result)

            # Sonuçları tekrar form sayfasına gönder
            return render_template("form.html", prediction=prediction, f1=f1, f2=f2, f3=f3, f4=f4)

        except Exception as e:
            return render_template("form.html", error=str(e), f1=f1, f2=f2, f3=f3, f4=f4)

    return render_template("form.html")

# Render uyumlu port ayarı
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
