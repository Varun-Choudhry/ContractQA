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


def chunker(text, size, overlap, mode):
        if mode == "fixed":
            return fixed_size_chunker(text, size, overlap)
        return

def extract_content(uploaded_file, tool):
    if tool = "azure_markdown":
        return extract_azure_markdwon(uploaded_file)
    return        
        
    
def extract_azure_markdwon(body):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", body=body)
    result = poller.result()
    return result.as_dict()
    
def get_embedding(text:str):
    #placeholder
    return ""

def fixed_size_chunker(text , chunk_size: int = 256, overlap: int = 20):    
    encoder = tiktoken.encoding_for_model("gpt-4o")
    token_text = encoder.encode(text)
    chunks = []
    encoder = tiktoken.encoding_for_model("gpt-4o")
    chunk_count = 1
    for i in range(0, len(token_text), chunk_size - overlap):
        chunk_text = encoder.decode(token_text[i:i+chunk_size])
        chunk_object = {
            "content": chunk_text ,
            "chunk_number": chunk_count,
            "_additional_": {"vector": get_embedding(chunk_text)}
        }
        chunks.append(chunk_object)
        chunk_count += 1
    return chunks
    