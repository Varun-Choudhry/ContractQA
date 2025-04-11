# llm_interaction/llm_handler.py

from core.llm.llm_client import LLMClient
from llm_interaction.prompt_builder import build_rag_prompt, build_system_prompt, build_agent_action_prompt, build_query_decomposition_prompt, build_final_answer_prompt
import re
import json
from typing import Dict, Any, List, Optional

class LLMHandler:
    def __init__(self, llm_client: LLMClient, chat_model: str):
        self.llm_client = llm_client
        self.chat_model = chat_model

    def generate_rag_response(self, query: str, context_chunks: list[str]) -> str:
        """
        Generates a RAG-based response using the LLM.

        Args:
            query: The user's search query.
            context_chunks: A list of relevant text chunks.

        Returns:
            The generated response string.
        """
        prompt = build_rag_prompt(query, context_chunks)
        system_prompt = build_system_prompt()
        response = self.llm_client.generate_text(prompt=prompt, model=self.chat_model, system_prompt=system_prompt)

        clean_text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return clean_text

    def decide_action(self, query: str, available_tools: Dict[str, Dict[str, Any]], previous_steps: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Decides which action to take based on the user query and available tools.

        Args:
            query: The current user query.
            available_tools: A dictionary describing the available tools.
            previous_steps: A list of previous (action, input, output).

        Returns:
            A dictionary containing the action to take and its parameters.
        """
        print("THIS IS BEFORE IN LLMHANDLER"+str(previous_steps))
        prompt = build_agent_action_prompt(query, available_tools, previous_steps)
        system_prompt = "" # You might need a system prompt for the agent
        response = self.llm_client.generate_text(prompt=prompt, model=self.chat_model, system_prompt=system_prompt)
        try:
            action_data = json.loads(response)
            return action_data
        except json.JSONDecodeError:
            print(f"Error decoding agent action response: {response}")
            return {"action": "error", "error": "Could not parse agent action."}

    def decompose_query(self, complex_query: str) -> List[str]:
        """
        Decomposes a complex query into a list of simpler sub-queries.
        """
        prompt = build_query_decomposition_prompt(complex_query)
        system_prompt = ""  # You might need a system prompt for decomposition
        response = self.llm_client.generate_text(
            prompt=prompt, model=self.chat_model, system_prompt=system_prompt
        )
        # Implement logic to parse the LLM response into a list of sub-queries
        sub_queries = self._parse_decomposition_response(response)
        return sub_queries

    def _parse_decomposition_response(self, llm_response: str) -> List[str]:
        """
        Parses the LLM's text response into a list of sub-queries.
        This will depend on how you prompt the LLM.
        """
        # Example parsing logic (adjust based on your LLM's output format)
        sub_queries = [line.strip() for line in llm_response.split('\n') if line.strip()]
        return sub_queries

    def generate_final_answer(self, query: str, context: Optional[str] = None, intermediate_results: Optional[List[Any]] = None) -> str:
        """
        Generates the final answer to the user's query.

        Args:
            query: The original user query.
            context: Relevant context if available.
            intermediate_results: Results from previous tool calls.

        Returns:
            The final answer string.
        """
        prompt = build_final_answer_prompt(query, context, intermediate_results)
        system_prompt = build_system_prompt() # You might want a specific system prompt for final answers
        response = self.llm_client.generate_text(prompt=prompt, model=self.chat_model, system_prompt=system_prompt)
        return response.strip()