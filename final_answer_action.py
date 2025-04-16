from pydantic import BaseModel
from typing import Optional

class FinalAnswerInputSchema(InputSchema):
    query: str
    provider: str
    context: list[str]    

class FinalAnswerOutputSchema(OutputSchema):
    result: str
    
class FinalAnswerAction(Action):
    
    def execute(self, schema: FinalAnswerInputSchema) -> FinalAnswerOutputSchema:
        return FinalAnswerOutputSchema(result=get_completion(schema.query,schema.provider,schema.context))
    