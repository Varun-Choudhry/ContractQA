from abc import ABC, abstractmethod
from openai import OpenAI

class LLMClient(ABC):
    @abstractmethod
    def embed_text(self, text: str, model: str) -> list[float]:
        pass

    @abstractmethod
    def generate_text(self, prompt: str, model: str, system_prompt: str = None) -> str:
        pass

    @abstractmethod
    def generate_text_tool(self, prompt: str, model: str, system_prompt: str = None) -> str:
        pass

