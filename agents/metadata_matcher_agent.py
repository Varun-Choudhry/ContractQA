### FILE: agents/metadata_matcher_agent.py

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase
#from schemas.metadata_matcher import MetadataMatcherInputSchema, MetadataMatcherOutputSchema
from core.llm.atomic_llm import get_llm_client
from pydantic import BaseModel
from typing import List, Optional
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

class MetadataMatcherInputSchema(BaseIOSchema):
    """Input schema for the Metadata Agent. Contains the user's message to be processed."""
    
    query: str

class MetadataMatcherOutputSchema(BaseIOSchema):
    """output schema for the Metadata Agent. """
    
    matches_metadata: bool
    matched_property: Optional[str] = None  # e.g., "page_numbers", "roles", etc.
    value: Optional[str] = None
    
metadata_matcher_agent = BaseAgent(
    BaseAgentConfig(
        client=get_llm_client(),
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
    background=[
        "You are a metadata matcher agent.",
        "Determine if a user query refers to any metadata fields in a document.",
        "Valid metadata fields are: 'page_number'.",
        "Your job is to check if the query mentions one of these fields."
    ],
    output_instructions=[
        "Respond with one object only.",
        "Return 'matches_metadata=True' if the query refers to a metadata field, and include 'matched_property' and 'value'.",
        "Do not return multiple tool calls or lists of results."
        "If it does NOT refer to metadata, return 'matches_metadata=False', and leave 'matched_property' and 'value' as None.",
        "Do NOT ask for clarification or additional context.",
        "Do NOT include anything outside of the response object."
    ]
)
,
        input_schema=MetadataMatcherInputSchema,
        output_schema=MetadataMatcherOutputSchema
    )
)
