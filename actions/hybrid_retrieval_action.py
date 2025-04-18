from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
import weaviate

class HybridRetrievalInputSchema(InputSchema):
    embeddings: list[list[float]]
    alpha: float = 0.3
    top_k: int = 5
    provider: str = "weaviate"
    chunks: list[str]        

class HybridRetrievalOutputSchema(OutputSchema):
    results: list[str]
    query: str
    
class HybridRetrievalAction(Action):
    InputSchema = HybridRetrievalInputSchema  
    OutputSchema = HybridRetrievalOutputSchema 
    
    def execute(self, schema: HybridRetrievalInputSchema) -> HybridRetrievalOutputSchema:
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.embeddings[0],schema.alpha,schema.top_k, schema.provider),query=schema.chunks[0])
    
    
def hybrid_search(embedding, alpha,top_k, provider):
    if provider=="weaviate":
        return hybrid_search_weaviate(embedding, alpha,top_k)
    return

def hybrid_search_weaviate(embedding, alpha,top_k):
    client = weaviate.connect_to_local()
    collection = client.collections.get("Document1")
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k)
    results = [obj.properties.get("body", "") for obj in response.objects]            
    return results    
    
    