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
    def __init__(self, config, mode):
        self.config = config.get("vector_db")
        self.mode = mode
    
    
    def execute(self, schema: HybridRetrievalInputSchema) -> HybridRetrievalOutputSchema:
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.embeddings[0], self.mode,self.config.get(self.mode)),query=schema.chunks[0])
    
    
def hybrid_search(embedding, provider, config):
    if provider=="weaviatelocal":
        return hybrid_search_weaviate(embedding, config.get('alpha'),config.get('top_k'), config)
    return

def hybrid_search_weaviate(embedding, alpha,top_k, config):
    client = weaviate.connect_to_local()
    collection = client.collections.get(config.get('collection'))
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k)
    results = [obj.properties.get("body", "") for obj in response.objects]            
    client.close()
    return results    
    
    