from pydantic import BaseModel
from typing import Optional
from src.rag_actions.actions.action import Action, InputSchema, OutputSchema
from src.rag_actions.retrieval.retrieval import hybrid_search

class HybridRetrievalInputSchema(InputSchema):
    embeddings: list[list[float]]
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
        return HybridRetrievalOutputSchema(results=hybrid_search(schema.chunks[0],schema.embeddings[0], self.mode,self.config.get(self.mode)),query=schema.chunks[0])
    
    

    