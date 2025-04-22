from openai import AzureOpenAI



def get_embedding_batch(chunks, mode, config):
    print("Provider for embeddings:"+ mode)
    if mode == "azure":
        return get_embedding_batch_azure(chunks, config)
    return    
        
def get_embedding_batch_azure(texts, config):
    
    print(f"Calling Azure OpenAI embedding API with model: {config.get("model")}")
    all_embeddings = []
   
    client = AzureOpenAI(
            api_version=config.get("api_version"),
            azure_endpoint=config.get("endpoint"),
            api_key=config.get("api_key")
        )
     
        
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        response = client.embeddings.create(input=batch, model=config.get("model"))
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    print(type(all_embeddings))
    return all_embeddings        