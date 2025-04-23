from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
from db.upload import upload_vectors


class VectorUploadInputSchema(InputSchema):
    embeddings: list[list[float]]
    chunks: list[str]
    metadata: dict = []
     

class VectorUploadOutputSchema(OutputSchema):
    result: bool
    
class VectorUploadAction(Action):
    InputSchema = VectorUploadInputSchema  
    OutputSchema = VectorUploadOutputSchema 
    def __init__(self, config, mode):
        self.config = config.get("vector_db")
        self.mode = mode
    
    def execute(self, schema: VectorUploadInputSchema) -> VectorUploadOutputSchema:
        return VectorUploadOutputSchema(result=upload_vectors(schema.embeddings,self.mode,schema.metadata, schema.chunks, self.config.get(self.mode)))
    
    
