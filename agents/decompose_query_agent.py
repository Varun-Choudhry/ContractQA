### FILE: agents/decompose_query_agent.py

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
#from schemas.decompose import DecomposeInputSchema, DecomposeOutputSchema
from core.llm.atomic_llm import get_llm_client
from pydantic import Field

from pydantic import BaseModel
from typing import List

class DecomposeInputSchema(BaseIOSchema):
    """Input schema for the Decompose Agent. Contains the user's message to be processed."""
    query: str = Field(..., description="The user's input message to be analyzed and responded to.")

class DecomposeOutputSchema(BaseIOSchema):
    """Output schema for the Decompose Agent. Contains the subqueries."""
    subqueries: List[str] = Field(..., description="The user's input message to be analyzed and responded to.")

decompose_query_agent = BaseAgent(
    BaseAgentConfig(
        client=get_llm_client(),
        model="gpt-4o-mini",
        system_prompt_generator = SystemPromptGenerator(
    background=[
                "You are a query decomposition agent for a document QA system.",
                "Your job is to break down complex queries **only if necessary** to improve retrieval and answering.",
                "If the query is already focused and specific, return it as-is in a single subquery.",
                "Only decompose if multiple **distinct aspects** need to be retrieved or reasoned over.",
                "Avoid overly generic or redundant subquestions.",
                "Prefer fewer, higher-quality subqueries."
    ],
    output_instructions=[
                "Always return a single list of subqueries, in one tool call.",
                "Do not return multiple separate tool responses.",
                "Use proper JSON format according to the schema."
    ]
),
        input_schema=DecomposeInputSchema,
        output_schema=DecomposeOutputSchema
    )
)

