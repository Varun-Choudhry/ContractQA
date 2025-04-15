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
        elif mode == "semantic"
            return breakpoint_semantic_chunker(text)
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

def get_embedding_batch(texts: list):
    #placeholder
    return [""]

#Need to decouple weaviate data object into a separate function
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


# Current implementation is very basic and inefficient. Will consider a two pointer approach to improve it.
# Need to identify a way to derive an acceptable cosine similarity threshold from the document itself for improved performance
# Since the data has been broken down into sentences, could possibly add sentence overlapping 
#PENDING TESTING
def breakpoint_semantic_chunker(text):
    from sklearn.metrics.pairwise import cosine_similarity
    chunks = []
    sentences = sentence_split(text)
    vectors = get_embedding_batch(sentences)
    current_chunk = [sentences[0]]
    for i in range(1,len(sentences)):
        similarity = cosine_similarity([vectors[i-1]], [vectors[i]])
        if similarity[0][0] <= 0.6:
            chunks.append(" ".join(current_chunk))
            current_chunk=[sentences[i]]
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))                    
    return chunks
    
#Using a basic sentence split using regex for now   
def sentence_split(text: str):
    sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]
    
    
    