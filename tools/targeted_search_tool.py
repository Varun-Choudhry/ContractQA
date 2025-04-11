from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from typing import List, Union
from pydantic import BaseModel
from atomic_agents.agents.base_agent import BaseIOSchema


# Input/output schemas
class TargetedSearchToolInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    query: str
    metadata_key: str
    metadata_value: Union[str, int, List[Union[str, int]]]
    top_k: int = 5

class TargetedSearchToolOutputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    results: List[str]

# Tool definition
class TargetedSearchTool(BaseTool):
    name = "targeted_search_tool"
    description = "Performs a filtered search based on metadata fields (e.g., page number, filename)."
    input_schema = TargetedSearchToolInputSchema
    output_schema = TargetedSearchToolOutputSchema

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, input: TargetedSearchToolInputSchema) -> TargetedSearchToolOutputSchema:
        chunks = self.retriever.targeted_search(
            query=input.query,
            metadata_key=input.metadata_key,
            metadata_value=input.metadata_value,
            top_k=input.top_k,
        )
        return TargetedSearchToolOutputSchema(results=[c.properties["content"] for c in chunks])
