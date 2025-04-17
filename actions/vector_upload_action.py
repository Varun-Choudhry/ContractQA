from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
import weaviate


class VectorUploadInputSchema(InputSchema):
    embeddings: list[list[float]]
    provider: str = "weaviate"
    metadata: dict = []
     

class VectorUploadOutputSchema(OutputSchema):
    result: bool
    
class VectorUploadAction(Action):
    InputSchema = VectorUploadInputSchema  
    OutputSchema = VectorUploadOutputSchema 
    
    def execute(self, schema: VectorUploadInputSchema) -> VectorUploadOutputSchema:
        return VectorUploadOutputSchema(result=upload_vectors(schema.embeddings,schema.provider,schema.metadata))
    
    
def upload_vectors(embeddings, provider, metadata):
    if provider == "weaviate":
        return upload_to_weaviate(embeddings, metadata)
    return    
        
        
def upload_to_weaviate(embeddings, metadata):
    client = weaviate.connect_to_local()
    collection = client.collections.get("Document")
    with collection.batch.dynamic() as batch:
        for embedding in embeddings:
            batch.add_object(
                vector=embedding
            )    
    return True