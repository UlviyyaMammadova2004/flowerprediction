# database.py
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://mammadovaulia82:Jungkook_19971997@cluster0.2g47tbp.mongodb.net/tasklist"
client = MongoClient(MONGO_URI)

db = client["predictions"]
collection = db["predicts"]


