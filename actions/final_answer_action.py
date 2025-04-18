from pydantic import BaseModel
from typing import Optional
from actions.action import Action, InputSchema, OutputSchema
from openai import AzureOpenAI

class FinalAnswerInputSchema(InputSchema):
    query: str
    provider: str = "azure"
    results: list[str]    

class FinalAnswerOutputSchema(OutputSchema):
    result: str
    
class FinalAnswerAction(Action):
    system_prompt = "You are a final answer generator. You task is to take the query and context,and generate a precise answer using ONLY the context. Reconcile any redunant terms present in the context"
    InputSchema = FinalAnswerInputSchema  
    OutputSchema = FinalAnswerOutputSchema 
    
    def execute(self, schema: FinalAnswerInputSchema) -> FinalAnswerOutputSchema:
        return FinalAnswerOutputSchema(result=get_completion(schema.query,schema.provider,schema.results,self.system_prompt))


def get_completion(query, provider, context, system_prompt):
    if provider == "azure":
        return get_completion_azure(query, context,system_prompt)
        
    return

def get_completion_azure(query, context, system_prompt):
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"Context:\n{'\n'.join(context)}\n\nQuery: {query}"})
    response = client.chat.completions.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content.strip()        
    