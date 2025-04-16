from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any

class InputSchema(BaseModel):
    pass  # Empty for now; can be extended for each agent's specific input

class OutputSchema(BaseModel):
    pass  # Empty for now; can be extended for each agent's specific output

class Action(ABC):
    @abstractmethod
    def execute(self, input_data: InputSchema) -> OutputSchema:
        pass
