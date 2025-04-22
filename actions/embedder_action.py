from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
from llm.embedding import get_embedding_batch

class EmbedderInputSchema(InputSchema):
    chunks: list[str]
     

class EmbedderOutputSchema(OutputSchema):
    embeddings: list[list[float]]  #or json depending on the metadata along with chunk
    chunks: list[str]
class EmbedderAction(Action):
    InputSchema = EmbedderInputSchema  
    OutputSchema = EmbedderOutputSchema 
    def __init__(self, config, mode):
        self.config = config.get("embedder")
        self.mode = mode
    
    def execute(self, schema: EmbedderInputSchema) -> EmbedderOutputSchema:
        return EmbedderOutputSchema(embeddings=get_embedding_batch(schema.chunks,self.mode,self.config.get(self.mode)),chunks=schema.chunks)
    