from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from core.llm.atomic_llm import get_llm_client
from pydantic import Field
from typing import List, Dict, Any

class EntityAgentInputSchema(BaseIOSchema):
    '''Input schema for entity extraction'''
    chunk: str = Field(..., description="The current chunk of contract text.")
    context_so_far: List[str] = Field(..., description="The previous chunk for context.")
    insights: List[Dict[str, Any]] = Field(..., description="The insight file")

class EntityAgentOutputSchema(BaseIOSchema):
    '''Output schema for extracted contract entities'''
    parties: List[Dict[str, str]]              # Each item has 'name' and 'contact_info'
    dates_and_durations: List[Dict[str, str]]  # Each item has 'value' and 'context'
    monetary_values: List[Dict[str, str]]      # Each item has 'value' and 'context'
    obligated_actions: List[str]

entity_agent = BaseAgent(
    BaseAgentConfig(
        client=get_llm_client(),
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an expert contract analyst focused on identifying key entities from contract documents.",
                "Use the current chunk and <Previous Chunk> to extract new or updated insights.",
                "Preserve all existing information in <Insights>. Do not remove or overwrite, only add new or updated entities."
            ],
            output_instructions=[
                "Extract structured insights, including: parties, monetary values, durations, and obligations.",
                "Do not duplicate previously identified entities from <Insights>. Only add or modify based on the current and prior chunks.",
                "For 'parties', output a dictionary with 'name' (e.g., 'Acme Corporation') and 'contact_info' (e.g., 'acme@example.com, 123 Main St'). If no contact info is available, leave it empty.",
                "For 'obligated_actions', rephrase them into complete, standalone sentences that would make sense without additional context. Avoid duplicates but ensure completeness.",
                "For 'monetary_values', output as a dictionary with 'value' (e.g., '$10,000') and 'context' (e.g., 'Monthly service fee').",
                "For 'dates_and_durations', output as a dictionary with 'value' (e.g., 'January 1, 2023 – December 31, 2023') and 'context' (e.g., 'Contract duration').",
                "Always ensure that new or updated information is added to <Insights>, but previous information is not lost."
            ]
        ),
        input_schema=EntityAgentInputSchema,
        output_schema=EntityAgentOutputSchema
    )
)

