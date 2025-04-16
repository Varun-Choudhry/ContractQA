from pydantic import BaseModel
from typing import Optional

class VectorUploadInputSchema(InputSchema):
    embeddings: list[list[float]]
    provider: str
    metadata: dict
     

class VectorUploadOutputSchema(OutputSchema):
    result: bool
    
class VectorUploadAction(Action):
    
    def execute(self, schema: VectorUploadInputSchema) -> VectorUploadOutputSchema:
        return VectorUploadOutputSchema(result=upload_vectors(schema.embeddings,schema.provider,schema.metadata))
    