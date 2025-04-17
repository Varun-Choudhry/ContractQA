import re
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken    
def chunk_text(text, size=0, overlap=0, mode='semantic'):
        if mode == "fixed":
            return fixed_size_chunker(text, size, overlap)
        elif mode == "semantic":
            return breakpoint_semantic_chunker(text)
        return

#Need to decouple weaviate data object into a separate function
def fixed_size_chunker(text , chunk_size: int = 256, overlap: int = 20):    
    encoder = tiktoken.encoding_for_model("gpt-4o")
    token_text = encoder.encode(text)
    chunks = []
    encoder = tiktoken.encoding_for_model("gpt-4o")
    chunk_count = 1
    for i in range(0, len(token_text), chunk_size - overlap):
        chunk_text = encoder.decode(token_text[i:i+chunk_size])
       
        chunks.append(chunk_text)
        chunk_count += 1
    return chunks

# Current implementation is very basic and inefficient. Will consider a two pointer approach to improve it.
# Need to identify a way to derive an acceptable cosine similarity threshold from the document itself for improved performance
# Since the data has been broken down into sentences, could possibly add sentence overlapping 
#PENDING TESTING
def breakpoint_semantic_chunker(text):
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
    
#Potential chunking implementation which leverages the insights provided by Azure Document Intelligence to create chunks which could better preserve logical groupings
def azure_chunker():
        return

#Azure DI has a dict of keys paragraphs, sections and tables, in context of this service paragraphs can also mean words, its the smallest unit of classification in Azure DI when it comes to how sections are mapped. Secttions have a list called elements which represent references to other sections/paragraphs as strings like "/sections/10", "/paragraphs/25" or "/tables/3". It also returns the whole document as a string
#Each of these groupings have an offset and length attached to them, which could be used to simple split the text string into groups. For example, chunk 1 could be text[<offset of section1>:offset+ length]. Due to the hiearchical nature of sections, need to find a better way which avoids redundancies.  
#possible solution, there are some 'paragraphs' which have a role in them, like title, section heading, the offset of these paragraphs could be use to as a breakpoint without having to traverse sections which is closer to a tree and would be difficult to flatten to something suitable for vector embeddings without making it unneccesarily complex.
#Need to analyze how tables structures would fit into this
