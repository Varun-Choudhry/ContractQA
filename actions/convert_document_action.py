from pydantic import BaseModel
from typing import Optional, Union
from actions.action import Action, InputSchema, OutputSchema
import io
from document.document_loader import convert_document 
class ConvertDocumentInputSchema(InputSchema):
    document: Union[str, bytes, io.BytesIO]
    class Config:
        arbitrary_types_allowed = True
   
    
class ConvertDocumentOutputSchema(OutputSchema):
    content: str 
    
class ConvertDocumentAction(Action):
    InputSchema = ConvertDocumentInputSchema  
    OutputSchema = ConvertDocumentOutputSchema 
    def __init__(self, config, mode):
        self.config = config.get("document")
        self.mode = mode
            
    def execute(self, schema: ConvertDocumentInputSchema) -> ConvertDocumentOutputSchema:
        return ConvertDocumentOutputSchema(content=convert_document(schema.document, self.mode, self.config.get(mode)))
    
