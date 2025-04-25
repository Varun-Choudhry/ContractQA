from pydantic import BaseModel
from typing import Optional
from src.rag_actions.actions.action import Action, InputSchema, OutputSchema
from openai import AzureOpenAI
from src.rag_actions.llm.llm import get_completion

class FinalAnswerInputSchema(InputSchema):
    query: str
    results: list[str]    

class FinalAnswerOutputSchema(OutputSchema):
    result: str
    
class FinalAnswerAction(Action):
    system_prompt = "You are a final answer generator. You task is to take the query and context,and generate a precise answer using ONLY the context. Reconcile any redunant terms present in the context"
    InputSchema = FinalAnswerInputSchema  
    OutputSchema = FinalAnswerOutputSchema 
    def __init__(self, config, mode):
        self.config = config.get("llm")
        self.mode = mode

    def execute(self, schema: FinalAnswerInputSchema) -> FinalAnswerOutputSchema:
        return FinalAnswerOutputSchema(result=get_completion(schema.query,self.mode,schema.results,self.system_prompt,self.config.get(self.mode)))

