from pydantic import root_validator
from typing import Optional
from chunker.chunker import chunk_text
from actions.action import Action, InputSchema, OutputSchema

class ChunkerInputSchema(InputSchema):
    content: str
    mode: str = "fixed"
    chunk_size: Optional[int] = 250
    overlap: Optional[int] = 20
    
  
class ChunkerOutputSchema(OutputSchema):
    chunks: list[str]  #or json depending on the metadata along with chunk
    
class ChunkerAction(Action):
    
    
    InputSchema = ChunkerInputSchema  # Ensure InputSchema is assigned
    OutputSchema = ChunkerOutputSchema  # Same for OutputSchema
 
    def __init__(self, config=None):
        self.config = config  
        
    def execute(self, schema: ChunkerInputSchema) -> ChunkerOutputSchema:
        return ChunkerOutputSchema(chunks=chunk_text(schema.content,schema.chunk_size,schema.overlap,schema.mode))
    