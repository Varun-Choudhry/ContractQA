#from typing import list
from core.vector_database.vector_db_client import VectorDBClient
from core.llm.llm_client import LLMClient
from weaviate.classes.query import Filter  # Make sure Filter is imported here
from typing import Union, List
class Retriever:
    def __init__(self, vector_db_client: VectorDBClient, llm_client: LLMClient, embedding_model: str):
        self.vector_db_client = vector_db_client
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.collection_name = "Document"  # Assuming your collection name

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, alpha: float = 0.0, filters: Filter = None):
        
        """
        Retrieves relevant document chunks based on a user query using hybrid search.

        Args:
            query: The user's search query.
            top_k: The number of top results to retrieve.
            alpha: The weight for the semantic vs. keyword search (0.0 for pure semantic, 1.0 for pure keyword).
            filters: Optional filter to apply to the search.

        Returns:
            A list of retrieved document objects from the vector database.
        """
        results = self.vector_db_client.hybrid_search(
            collection_name=self.collection_name,
            query=query,
            alpha=alpha,
            limit=top_k,
            filters=filters  # Pass the filters to the hybrid_search method
        )
        return results

    def retrieve_by_id(self, object_id: str):
        """
        Retrieves a specific object from the vector database by its ID.
        (Note: You might need to adapt this based on your Weaviate client's capabilities)
        """
        # This is a placeholder as direct retrieval by ID might vary
        # depending on your VectorDBClient implementation.
        raise NotImplementedError("Retrieval by ID is not yet implemented in this example.")


    def retrieve_by_metadata_filter(self, query: str, filter_property: str, filter_value: str, top_k: int = 5, alpha: float = 0.0):

        results = self.vector_db_client.hybrid_search(
            collection_name=self.collection_name,
            query=query,
            filters=Filter.by_property(filter_property).equal(filter_value),
            alpha=alpha,
            limit=top_k,
        )
        return results

    def retrieve_with_metadata_filter(self, query: str, metadata_filter: dict[str, any], top_k: int = 5, alpha: float = 0.0):
        """Retrieves relevant chunks with multiple metadata filters."""
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

        results = self.vector_db_client.hybrid_search(
            collection_name=self.collection_name,
            query=query,
            alpha=alpha,
            limit=top_k,
            filters=compound_filter
        )
        return results
        
    def hybrid_search(self, query: str, top_k: int = 5):
        """
        Retrieves relevant document chunks based on a user query using hybrid search.

        Args:
            query: The user's search query.
            top_k: The number of top results to retrieve.
            alpha: The weight for the semantic vs. keyword search (0.0 for pure semantic, 1.0 for pure keyword).
            filters: Optional filter to apply to the search.

        Returns:
            A list of retrieved document objects from the vector database.
        """
        results = self.vector_db_client.hybrid_search(
            collection_name=self.collection_name,
            query=query,
            alpha=0.3,
            limit=top_k,
           # Pass the filters to the hybrid_search method
        )
        return results        
        
    def targeted_search(self, query: str, top_k: int = 5, key: str = '', value: Union[str, int, List[Union[str, int]]] = None):
        """
        Retrieves relevant document chunks based on a user query using hybrid search.

        Args:
            query: The user's search query.
            top_k: The number of top results to retrieve.
            alpha: The weight for the semantic vs. keyword search (0.0 for pure semantic, 1.0 for pure keyword).
            filters: Optional filter to apply to the search.

        Returns:
            A list of retrieved document objects from the vector database.
        """
        if not isinstance(value, list):
            value = [value]        
        filters=Filter.by_property(key).containsAny(value)
        results = self.vector_db_client.hybrid_search(
            collection_name=self.collection_name,
            query=query,
            alpha=0.3,
            limit=top_k,
           # Pass the filters to the hybrid_search method
        )
        return results         