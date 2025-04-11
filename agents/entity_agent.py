from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from core.llm.atomic_llm import get_llm_client
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EntityAgentInputSchema(BaseIOSchema):
    '''Hello '''
    chunk: str = Field(..., description="The current chunk of contract text.")
    context_so_far: List[str] = Field(..., description="The previous 3 chunks for context.")
    prior_insights: List[Dict[str, Any]] = Field(..., description="Previous entity insights for reference.")

class EntityAgentOutputSchema(BaseIOSchema):
    '''Hello'''
    parties: List[str]
    dates_and_durations: List[str]
    monetary_values: List[str]
    obligated_actions: List[str]

entity_agent = BaseAgent(
    BaseAgentConfig(
        client=get_llm_client(),
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an expert contract analyst focused on identifying key entities from contract documents.",
                "Use the current chunk and recent context to extract precise, structured insights.",
                "Pay close attention to the 'Context So Far' and 'Prior Insights' to avoid adding redundant information.",
                "Focus on identifying new entities or significant updates to previously identified entities."
            ],
            output_instructions=[
                "Extract a list of parties, monetary values, durations, and obligations.",
                "Output a dictionary with keys: parties, dates_and_durations, monetary_values, obligated_actions.",
                "Only include entities clearly stated or reasonably inferred from the current and prior chunks that have not been mentioned identically before.",
                "Be mindful of variations in entity names (e.g., 'Acme Corp.' vs. 'Acme Corporation') but only add a new entry if it represents a distinct entity or a significant new detail.",
                "When listing obligations, focus on the core action and avoid repeating the same obligation if mentioned in slightly different phrasing across chunks."
            ]
        ),
        input_schema=EntityAgentInputSchema,
        output_schema=EntityAgentOutputSchema
    )
)
