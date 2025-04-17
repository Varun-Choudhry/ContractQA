from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema


class FinalAnswerInputSchema(InputSchema):
    query: str
    provider: str
    context: list[str]    

class FinalAnswerOutputSchema(OutputSchema):
    result: str
    
class FinalAnswerAction(Action):
    InputSchema = FinalAnswerInputSchema  
    OutputSchema = FinalAnswerOutputSchema 
    
    def execute(self, schema: FinalAnswerInputSchema) -> FinalAnswerOutputSchema:
        return FinalAnswerOutputSchema(result=get_completion(schema.query,schema.provider,schema.context))
    