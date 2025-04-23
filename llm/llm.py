from openai import AzureOpenAI

def get_completion(query, provider, context, system_prompt, config):
    if provider == "azure":
        return get_completion_azure(query, context,system_prompt, config)
        
    return

def get_completion_azure(query, context, system_prompt, config):
    client = AzureOpenAI(
        api_version=config.get("api_version"),
            azure_endpoint=config.get("endpoint"),
            api_key=config.get("api_key")
    )

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"Context:\n{'\n'.join(context)}\n\nQuery: {query}"})
    response = client.chat.completions.create(model=config.get("model"), messages=messages)
    return response.choices[0].message.content.strip()        
    