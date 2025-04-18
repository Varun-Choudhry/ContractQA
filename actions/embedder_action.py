from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
from llm.embedding import get_embedding_batch

class EmbedderInputSchema(InputSchema):
    chunks: list[str]
    provider: str = "azure"
    model: str = "text-embedding-3-large"
     

class EmbedderOutputSchema(OutputSchema):
    embeddings: list[list[float]]  #or json depending on the metadata along with chunk
    chunks: list[str]
class EmbedderAction(Action):
    InputSchema = EmbedderInputSchema  
    OutputSchema = EmbedderOutputSchema 
    
    def execute(self, schema: EmbedderInputSchema) -> EmbedderOutputSchema:
        return EmbedderOutputSchema(embeddings=get_embedding_batch(schema.chunks,schema.provider,schema.model),chunks=schema.chunks)
    