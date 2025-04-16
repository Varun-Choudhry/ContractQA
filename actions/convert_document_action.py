from pydantic import BaseModel
from typing import Optional, Union
from actions.action import Action
from actions.action import Action, InputSchema, OutputSchema
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
import io

class ConvertDocumentInputSchema(InputSchema):
    mode: str
    document: Union[str, bytes, io.BytesIO]
    class Config:
        arbitrary_types_allowed = True
   
    
class ConvertDocumentOutputSchema(OutputSchema):
    content: str 
    
class ConvertDocumentAction(Action):
    
    def execute(self, schema: ConvertDocumentInputSchema) -> ConvertDocumentOutputSchema:
        return ConvertDocumentOutputSchema(content=convert_document(schema.document, schema.mode))
    
##added here for testing, will put in a different file
    
def convert_document(document, mode: str) -> str:
    if mode == "azure":
        return convert_with_azure(document)
    elif mode == "docling":
        return convert_with_docling(document)
        
        
def convert_with_docling(document):
    source = DocumentStream(name="my_doc.pdf", stream=document)
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()