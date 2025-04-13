from pymongo import MongoClient
from config.config import config
from datetime import datetime

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(config["mongo_uri"])
        self.db = self.client[config["mongo_db"]]
        self.collection = self.db[config["mongo_collection"]]

    def insert_document(self, doc):
        self.collection.update_one(
            {"filename": doc["filename"]},
            {"$setOnInsert": {**doc, "uploaded_at": datetime.utcnow(), "status": "uploaded"}},
            upsert=True
        )

    def get_document_by_filename(self, filename):
        return self.collection.find_one({"filename": filename})

    def document_exists(self, filename):
        return self.collection.count_documents({"filename": filename}, limit=1) > 0

    def get_all_filenames(self):
        return [doc["filename"] for doc in self.collection.find({}, {"filename": 1, "_id": 0})]

    def get_all_documents_with_status(self):
        return list(self.collection.find({}, {"filename": 1, "status": 1, "_id": 0}))

    def update_document_status(self, filename, status):
        self.collection.update_one(
            {"filename": filename},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )

    def update_document_insight(self, filename, json_insight):
        self.collection.update_one(
            {"filename": filename},
            {"$set": {"json_insight": json_insight, "summarized_at": datetime.utcnow(), "status": "summarized"}},
            upsert=True
        )

    def get_document_insight(self, filename):
        doc = self.collection.find_one({"filename": filename}, {"json_insight": 1, "_id": 0})
        return doc