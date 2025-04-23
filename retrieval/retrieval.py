import weaviate


def hybrid_search(embedding, provider, config):
    if provider=="weaviatelocal":
        return hybrid_search_weaviate(embedding, config.get('alpha'),config.get('top_k'), config)
    return

def hybrid_search_weaviate(embedding, alpha,top_k, config):
    client = weaviate.connect_to_local()
    collection = client.collections.get(config.get('collection'))
    response = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k)
    results = [obj.properties.get("body", "") for obj in response.objects]            
    client.close()
    return results    
    