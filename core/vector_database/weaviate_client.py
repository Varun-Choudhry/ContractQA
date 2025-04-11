#core/vector_database/weaviate_client.py

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from typing import List, Dict, Any

from core.vector_database.vector_db_client import VectorDBClient
from weaviate.classes.query import Filter

class WeaviateClient(VectorDBClient):
    def __init__(self):
        self.client = None

    def connect(self, url: str, headers: Dict[str, str] = None):
        self.client = weaviate.connect_to_local(headers=headers)

    def check_collection_exists(self, collection_name: str):
        return self.client.collections.exists(collection_name)


    def create_collection(self, collection_name: str, vector_index_config: Dict[str, Any], vectorizer_config: Dict[str, Any], properties: List[Dict[str, Any]]):
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=vectorizer_config,
            vector_index_config=vector_index_config,
            properties=properties
        )

    def get_collection(self, collection_name: str):
        return self.client.collections.get(collection_name)

    def add_data_objects(self, collection_name: str, data_objects: List[Dict[str, Any]]):
        collection = self.client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for data_object in data_objects:
                batch.add_object(
                    properties={k: v for k, v in data_object.items() if k != "_additional"},
                    vector=data_object["_additional"]["vector"]
                )

    def hybrid_search(self, collection_name: str, query: str, alpha: float, limit: int, filters: Filter = None):
        collection = self.client.collections.get(collection_name)
        return collection.query.hybrid(query=query, alpha=alpha, limit=limit, filters=filters).objects

    def delete_all_collections(self):
        """Deletes all collections in the Weaviate instance."""
        if self.client:
            for collection in self.client.collections.list().keys():
                print(f"Deleting collection: {collection}")
                self.client.collections.delete(collection)
            print("✅ All collections deleted.")
        else:
            print("⚠️ Weaviate client is not connected.")

    def retrieve_with_metadata_filter(self, collection_name: str, query: str, metadata_filter: Dict[str, Any], top_k: int = 5, alpha: float = 0.0):
        """Retrieves objects with a complex metadata filter."""
        collection = self.client.collections.get(collection_name)
        compound_filter = None
        if metadata_filter:
            conditions = []
            for prop, value in metadata_filter.items():
                conditions.append(Filter.by_property(prop).equal(value))

            if conditions:
                if len(conditions) > 1:
                    compound_filter = Filter.by_operator("and", conditions=conditions)
                else:
                    compound_filter = conditions[0]

        return self.hybrid_search(collection_name, query, alpha, top_k, filters=compound_filter)