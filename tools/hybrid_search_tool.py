from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from typing import List
from pydantic import BaseModel
from atomic_agents.agents.base_agent import BaseIOSchema
# Input/output schemas
class HybridSearchToolInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""
    
    query: str
    top_k: int = 5

class HybridSearchToolOutputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""
    
    results: List[str]

# Tool definition
class HybridSearchTool(BaseTool):
    name = "hybrid_search_tool"
    description = "Performs a hybrid vector+keyword search over document chunks."
    input_schema = HybridSearchToolInputSchema
    output_schema = HybridSearchToolOutputSchema

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, input: HybridSearchToolInputSchema) -> HybridSearchToolOutputSchema:
        chunks = self.retriever.hybrid_search(query=input.query, top_k=input.top_k)
        print(str(chunks))
        return HybridSearchToolOutputSchema(results=[c.properties["content"] for c in chunks])
