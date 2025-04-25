from pydantic import root_validator
from typing import Optional
from src.rag_actions.chunker.chunker import chunk_text
from src.rag_actions.actions.action import Action, InputSchema, OutputSchema

class ChunkerInputSchema(InputSchema):
    content: str
    
  
class ChunkerOutputSchema(OutputSchema):
    chunks: list[str]  #or json depending on the metadata along with chunk
    
class ChunkerAction(Action):
    
    
    InputSchema = ChunkerInputSchema  # Ensure InputSchema is assigned
    OutputSchema = ChunkerOutputSchema  # Same for OutputSchema
 
    def __init__(self, config, mode):
        self.config = config.get("chunker")
        self.mode = mode
        
    def execute(self, schema: ChunkerInputSchema) -> ChunkerOutputSchema:
        return ChunkerOutputSchema(chunks=chunk_text(schema.content,self.config.get(self.mode),self.mode))
    