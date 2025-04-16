from pydantic import BaseModel
from typing import Optional

class EmbedderInputSchema(InputSchema):
    text: list[str]
    provider: str
    model: str
     

class EmbedderOutputSchema(OutputSchema):
    embeddings: list[list[float]]  #or json depending on the metadata along with chunk
    
class EmbedderAgent(Action):
    
    def execute(self, schema: EmbedderInputSchema) -> EmbedderOutputSchema:
        return EmbedderOutputSchema(embeddings=get_embedding_batch(schema.text,schema.provider,schema.model))
    