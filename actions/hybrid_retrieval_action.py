from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema


class HybridRetrievalInputSchema(InputSchema):
    embeddings: list[list[float]]
    alpha: float = 0.3
    top_k: int = 5
    provider: str
        

class HybridRetrievalOutputSchema(OutputSchema):
    results: list[str]
    
class HybridRetrievalAction(Action):
    InputSchema = HybridRetrievalInputSchema  
    OutputSchema = HybridRetrievalOutputSchema 
    
    def execute(self, schema: HybridRetrievalInputSchema) -> HybridRetrievalOutputSchema:
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.embeddings[0],schema.alpha,schema.top_k, schema.provider))
    