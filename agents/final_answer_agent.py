### FILE: agents/final_answer_agent.py

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

#from schemas.final_answer import FinalAnswerInputSchema, FinalAnswerOutputSchema
from core.llm.atomic_llm import get_llm_client
from pydantic import BaseModel
from typing import List

class FinalAnswerInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""
    query: str
    retrieved_chunks: List[str]  # Plain text or structured snippets

class FinalAnswerOutputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""
    
    answer: str


final_answer_agent = BaseAgent(
    BaseAgentConfig(
        client=get_llm_client(),
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a final answer generator agent.",
                "Your task is to take a query and supporting context from document search results and return a precise, concise answer."
            ],
            output_instructions=[
                "Summarize or synthesize the information to answer the query clearly."
            ]
        ),
        input_schema=FinalAnswerInputSchema,
        output_schema=FinalAnswerOutputSchema
    )
)
