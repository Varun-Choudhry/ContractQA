from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema


class HybridRetrievalInputSchema(InputSchema):
    embedding: list[float]
    alpha: float
    top_k: int
    provider: str
        

class HybridRetrievalOutputSchema(OutputSchema):
    results: list[str]
    
class HybridRetrievalAction(Action):
    InputSchema = HybridRetrievalInputSchema  
    OutputSchema = HybridRetrievalOutputSchema 
    
    def execute(self, schema: HybridRetrievalInputSchema) -> HybridRetrievalOutputSchema:
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.embedding,schema.alpha,schema.top_k, schema.provider))
    