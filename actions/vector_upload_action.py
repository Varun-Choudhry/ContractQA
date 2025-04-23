from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
import weaviate


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
    
    
def upload_vectors(embeddings, provider, metadata, chunks, config):
    if provider == "weaviatelocal":
        return upload_to_weaviate(embeddings, metadata, chunks, config)
    return    
        
        
def upload_to_weaviate(embeddings, metadata, chunks, config):
    client = weaviate.connect_to_local()
    collection = client.collections.get(config.get("collection"))
    with collection.batch.dynamic() as batch:
        for i in range(0,len(embeddings)):
            batch.add_object(
                properties={"body": chunks[i]},
                vector=embeddings[i]
            )    
    return True