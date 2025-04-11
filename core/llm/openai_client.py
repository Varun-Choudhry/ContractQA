from openai import OpenAI
from core.llm.llm_client import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_text(self, text: str, model: str) -> list[float]:
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    def generate_text(self, prompt: str, model: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()

    def generate_text_tool(self, prompt: str, model: str, system_prompt: str = None) -> str:
        return self.generate_text(prompt, model, system_prompt)
