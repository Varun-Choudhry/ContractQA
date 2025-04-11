from abc import ABC, abstractmethod
#from typing import list, dict, any

class VectorDBClient(ABC):
    @abstractmethod
    def connect(self, url: str, headers: dict[str, str] = None):
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, vector_index_config: dict[str, any], vectorizer_config: dict[str, any], properties: list[dict[str, any]]):
        pass

    @abstractmethod
    def get_collection(self, collection_name: str):
        pass

    @abstractmethod
    def add_data_objects(self, collection_name: str, data_objects: list[dict[str, any]]):
        pass

    @abstractmethod
    def hybrid_search(self, collection_name: str, query: str, vector: list[float], alpha: float, limit: int):
        pass

    @abstractmethod
    def delete_all_collections(self):
        pass