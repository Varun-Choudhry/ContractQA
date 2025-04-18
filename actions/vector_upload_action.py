from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
import weaviate


class VectorUploadInputSchema(InputSchema):
    embeddings: list[list[float]]
    chunks: list[str]
    provider: str = "weaviate"
    metadata: dict = []
     

class VectorUploadOutputSchema(OutputSchema):
    result: bool
    
class VectorUploadAction(Action):
    InputSchema = VectorUploadInputSchema  
    OutputSchema = VectorUploadOutputSchema 
    
    def execute(self, schema: VectorUploadInputSchema) -> VectorUploadOutputSchema:
        return VectorUploadOutputSchema(result=upload_vectors(schema.embeddings,schema.provider,schema.metadata, schema.chunks))
    
    
def upload_vectors(embeddings, provider, metadata, chunks):
    if provider == "weaviate":
        return upload_to_weaviate(embeddings, metadata, chunks)
    return    
        
        
def upload_to_weaviate(embeddings, metadata, chunks):
    client = weaviate.connect_to_local()
    collection = client.collections.get("Document1")
    with collection.batch.dynamic() as batch:
        for i in range(0,len(embeddings)):
            batch.add_object(
                properties={"body": chunks[i]},
                vector=embeddings[i]
            )    
    return True