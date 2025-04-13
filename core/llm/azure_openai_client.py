from openai import OpenAI
from core.llm.llm_client import LLMClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

class AzureOpenAIClient(LLMClient):
    def __init__(self, chat_version: str, chat_endpoint: str, chat_key: str, embedding_version: str, embedding_endpoint: str, embedding_key: str):
        self.chat_client = AzureOpenAI(
    api_version=chat_version,
    azure_endpoint=chat_endpoint,
    api_key=chat_key,
)
        self.embedding_client  = AzureOpenAI(
    api_version="",
    azure_endpoint="",
    api_key=""
)
    def embed_text(self, text: str, model: str) -> list[float]:
        print(f"Calling Azure OpenAI embedding API with model: {model}") # Add this for debugging

        response = self.embedding_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    def generate_text(self, prompt: str, model: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.chat_client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()

  
