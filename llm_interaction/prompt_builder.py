# llm_interaction/prompt_builder.py

def build_rag_prompt(query: str, context_chunks: list[str]) -> str:
    """
    Builds a prompt for the LLM using the user's query and retrieved context chunks.

    Args:
        query: The user's search query.
        context_chunks: A list of relevant text chunks retrieved from the vector database.

    Returns:
        A formatted prompt string.
    """
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"""
    return prompt

def build_system_prompt() -> str:
    """
    Builds the system prompt for the LLM.
    """
    return (
        "You are a helpful assistant answering questions about a contract. "
        "Use the provided context to answer the user’s question as clearly and accurately as possible."
    )

def build_query_decomposition_prompt(complex_query: str) -> str:
    prompt = f"""Decompose the following complex user query into a series of simpler, self-contained sub-queries that can be answered independently. Ensure each sub-query is clear and directly addresses a part of the original query.

    Complex Query: "{complex_query}"

    Sub-Queries:
    """
    return prompt

def format_previous_steps_for_prompt(previous_steps: list = None) -> str:
    """Formats the previous steps (list of dicts) into a readable string for the prompt."""
    if not previous_steps:
        return ""
    formatted_steps = "\n**Previous Steps:**\n"
    for i, step in enumerate(previous_steps):
        tool_name = step.get("action", "N/A")
        tool_input = step.get("input", {})
        tool_output = step.get("output", {})
        formatted_steps += f"Step {i+1}:\n"
        formatted_steps += f"- Tool: {tool_name}\n"
        formatted_steps += f"- Input: {tool_input}\n"
        formatted_steps += f"- Output: {tool_output}\n"
    return formatted_steps + "\n"
def build_agent_action_prompt(query: str, available_tools: dict, previous_steps: list = None) -> str:
    """
    Builds a prompt for the agent to decide which tool to use next, considering previous steps.

    Args:
        query: The current user query.
        available_tools: A dictionary describing the available tools (name, description, parameters).
        previous_steps: A list of (tool_name, tool_input, tool_output) tuples from previous actions.

    Returns:
        The prompt string.
    """
    prompt = f"""You are an intelligent agent designed to answer user questions about documents. You have access to the following tools:

    {format_available_tools(available_tools)}

    {format_previous_steps_for_prompt(previous_steps)}

    **Current User Query:** "{query}"

    **Thought Process:**

    1. **Review Previous Steps:** Consider the tools you have already used and their outputs. Avoid repeating the exact same tool with the exact same inputs unless absolutely necessary and you have a clear reason to believe it will yield new information.

    2. **Complexity Analysis:** First, analyze the user's query for complexity. If it involves multiple distinct questions or requires breaking down into smaller parts, consider using the 'decompose_query_tool'.

    3. **Metadata Identification:** After (optionally) decomposing the query, or if the query is simple, try to identify any specific metadata filters within the query that can be used for targeted searching. Use the 'analyze_query_for_metadata_tool' for this.

    4. **Information Retrieval:**
        - If specific metadata filters were identified, use the 'perform_targeted_search_tool' to retrieve relevant information.
        - If the query relates to a specific document (and no specific metadata is found), or as a default retrieval mechanism, use the 'perform_document_hybrid_search_tool' specifying the relevant filename.
        - For broader searches across the entire document database when no specific document is implied, use the 'perform_generic_hybrid_search_tool'.

    5. **Final Answer Generation:** Once you have gathered sufficient relevant information and have addressed the user's query, use the 'generate_final_answer_tool' to formulate your response.

    Decide which tool to use next, or if you have enough information to generate a final answer.

    If you decide to use a tool, respond in the following JSON format:
    ```json
    {{
        "action": "tool_name",
        "parameters": {{
            "parameter1": "value1",
            "parameter2": "value2",
            ...
        }}
    }}
    ```

    If you have enough information to generate a final answer, respond in the following JSON format:
    ```json
    {{
        "action": "final_answer",
        "answer": "your generated answer here"
    }}
    ```

    Remember to carefully consider the current query, the descriptions of the available tools, and **what you have already tried** to determine the most appropriate action. Avoid getting stuck in repetitive loops.
    """
    
    print("***Prompt****"+prompt)
    return prompt

def format_available_tools(tools: dict) -> str:
    tool_descriptions = []
    for name, details in tools.items():
        tool_descriptions.append(f"- **{name}**: {details['description']}")
        if 'parameters' in details:
            params = ", ".join([f"{p}: {t}" for p, t in details['parameters'].items()])
            tool_descriptions.append(f"  Parameters: {params}")
    return "\n".join(tool_descriptions)

def format_previous_steps(steps: list) -> str:
    if not steps:
        return "You haven't taken any steps yet."
    history = "Here are the results of your previous steps:\n"
    for i, step in enumerate(steps):
        history += f"**Step {i+1}:**\n"
        history += f"  Action: {step.get('action')}\n"
        history += f"  Input: {step.get('input')}\n"
        history += f"  Output: {step.get('output')}\n"
    return history

def build_final_answer_prompt(query: str, context: str = None, intermediate_results: list = None) -> str:
    """
    Builds a prompt for the agent to generate the final answer based on gathered information.

    Args:
        query: The original user query.
        context: Relevant text context (if any).
        intermediate_results: A list of results from previous tool calls.

    Returns:
        The prompt string.
    """
    prompt = f"""Based on the following information, generate a concise and informative answer to the user's query: "{query}"

    """
    if context:
        prompt += f"\nRelevant Context:\n{context}\n"
    if intermediate_results:
        prompt += "\nInformation from previous steps:\n"
        for i, result in enumerate(intermediate_results):
            prompt += f"Step {i+1}: {result}\n"

    prompt += "\nFinal Answer:"
    return prompt

def build_metadata_analysis_prompt(query: str, available_metadata_fields: list[str]) -> str:
    """
    Builds a prompt for the LLM to analyze a query and identify metadata filters,
    informed by the available metadata fields in the Weaviate schema.

    Args:
        query: The user's search query.
        available_metadata_fields: A list of strings representing the names of the
                                     metadata fields in the Weaviate schema
                                     that can be used for filtering.

    Returns:
        A prompt string instructing the LLM to extract metadata.
    """
    available_fields_str = ", ".join(available_metadata_fields)
    prompt = f"""Analyze the following user query to identify any specific metadata filters that correspond to the available metadata fields in our document database. The available metadata fields for filtering are: {available_fields_str}.

    Please identify the metadata field and its corresponding value. If a field is mentioned in the query and a value is provided or implied, extract it. If multiple filters are present, list them clearly. If no explicit metadata filters related to the available fields are found, indicate that.

    User Query: "{query}"

    Identified Metadata Filters (as a JSON object if possible, otherwise as field: value pairs):"""
    return prompt