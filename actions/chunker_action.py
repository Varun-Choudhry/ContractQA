from pydantic import BaseModel
from typing import Optional
from chunker.chunker import chunk_text
from actions.action import Action

class ChunkerInputSchema(InputSchema):
    text: str
    mode: str
    chunk_size: Optional[int]
    overlap: Optional[int]
    
    @root_validator
    def validate_schema(cls, values):
        mode = values.get('mode')
        chunk_size = values.get('chunk_size')
        if mode == "fixed" and not chunk_size:
            raise ValueError
        return values    

class ChunkerOutputSchema(OutputSchema):
    chunks: list[str]  #or json depending on the metadata along with chunk
    
class ChunkerAction(Action):
    
    def execute(self, schema: ChunkerInputSchema) -> ChunkerOutputSchema:
        return ChunkerOutputSchema(chunks=chunk_text(schema.text,schema.chunk_size,schema.overlap,schema.mode))
    