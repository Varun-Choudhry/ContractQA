from pydantic import BaseModel
from typing import Optional

class HybridRetrievalInputSchema(InputSchema):
    embedding: list[float]
    alpha: float
    top_k: int
    provider: str
        

class HybridRetrievalOutputSchema(OutputSchema):
    results: list[str]
    
class HybridRetrievalAction(Action):
    
    def execute(self, schema: HybridRetrievalInputSchema) -> HybridRetrievalOutputSchema:
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.embedding,schema.alpha,schema.top_k, schema.provider))
    