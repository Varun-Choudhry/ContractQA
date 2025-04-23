import weaviate

def upload_vectors(embeddings, provider, metadata, chunks, config):
    if provider == "weaviatelocal":
        return upload_to_weaviate(embeddings, metadata, chunks, config)
    return    
        
        
def upload_to_weaviate(embeddings, metadata, chunks, config):
    client = weaviate.connect_to_local()
    collection = client.collections.get(config.get("collection"))
    with collection.batch.dynamic() as batch:
        for i in range(0,len(embeddings)):
            batch.add_object(
                properties={"body": chunks[i]},
                vector=embeddings[i]
            )    
    return True