from pymongo import MongoClient

# Doğru URI
MONGO_URI = "mongodb+srv://mammadovaulia82:Jungkook_19971997@cluster0.2g47tbp.mongodb.net/tasklist"

client = MongoClient(MONGO_URI)

# DÜZELTİLDİ: doğru veritabanı ve koleksiyon
db = client["predictions"]
collection = db["predicts"]

