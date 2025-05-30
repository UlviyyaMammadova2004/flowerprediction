from pymongo import MongoClient

# MongoDB bağlantı URI'n (şifre özel karakter içermediği için doğrudan kullanılabilir)
MONGO_URI = "mongodb+srv://mammadovaulia82:Jungkook_19971997@cluster0.2g47tbp.mongodb.net/tasklist"

# Bağlantıyı oluştur
client = MongoClient(MONGO_URI)

# Veritabanı ve koleksiyon seçimi
db = client["flowerdb"]
collection = db["predictions"]
