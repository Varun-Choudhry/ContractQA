from openai import AzureOpenAI



def get_embedding_batch(chunks, provider, model):
    if provider == "azure":
        return get_embedding_batch_azure(chunks, model)
    return    
        
def get_embedding_batch_azure(texts: list[str], model: str):
    print(f"Calling Azure OpenAI embedding API with model: {model}")
    all_embeddings = []
    client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint="",
            api_key=""
        )
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings        