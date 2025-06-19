from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Veri setini yükle
data = load_iris()
X = data.data
y = data.target

# Modeli eğit
clf = RandomForestClassifier()
clf.fit(X, y)

# Modeli kaydet
joblib.dump(clf, "iris_model.pkl")

print("✅ Model başarıyla eğitildi ve iris_model.pkl dosyasına kaydedildi.")
