from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
from openai import AzureOpenAI

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
        return FinalAnswerOutputSchema(result=get_completion(schema.query,self.mode,schema.results,self.system_prompt,self.config.get(mode)))


def get_completion(query, provider, context, system_prompt, config):
    if provider == "azure":
        return get_completion_azure(query, context,system_prompt, config)
        
    return

def get_completion_azure(query, context, system_prompt, config):
    client = AzureOpenAI(
        api_version=config.get("api_version"),
            azure_endpoint=config.get("endpoint"),
            api_key=config.get("api_key")
    )

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"Context:\n{'\n'.join(context)}\n\nQuery: {query}"})
    response = client.chat.completions.create(model=config.get("model"), messages=messages)
    return response.choices[0].message.content.strip()        
    