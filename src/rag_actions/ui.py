#TODO add necessary imports
#Decouple everything into their domain folders
#handle secrets

##### UPLOAD TAB#####
#user gives document
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])

#extract content
text = extract_content(uploaded_file, "azure_markdown")



##### QA #####

#User gives query
chunks = chunker(text, size, overlap, "fixed")

#upload to vector db
upload_result = upload(chunks, "weaviate")

#call agent
    #retrieval if applicable

#return answer to user

############### Weaviate Start######################
def connect_weaviate(url, headers):
    client = weaviate.connect(url=url,headers=headers)

#Weaviate hybrid search
def weaviate_retrieve_hybrid(collection_name, query, top_k, alpha, filters):
    client= connect_weaviate(url, headers)
    collection = self.client.collections.get(collection_name)
    return collection.query.hybrid(query=query, alpha=alpha, limit=limit, filters=filters).objects

def add_data_objects_batch(collection_name, data_objects):
    client= connect_weaviate(url, headers)
    collection = client.collections.get(collection_name)
    with collection.batch.dynamic() as batch:
        for data_object in data_objects:
            batch.add_object(
                properties={k: v for k, v in data_object.items() if k != "_additional"},
                vector=data_object["_additional"]["vector"]
            )

############### Weaviate Ending######################


def extract_content(uploaded_file, tool):
    if tool = "azure_markdown":
        return extract_azure_markdwon(uploaded_file)
    return        
        
    
def extract_azure_markdwon(body):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", body=body)
    result = poller.result()
    return result.as_dict()
    


