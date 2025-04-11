import streamlit as st
from config.config import config
from core.llm.openai_client import OpenAIClient  # Import the specific client
from core.vector_database.weaviate_client import WeaviateClient
from core.document.document_loader import load_document_from_upload
from core.document.chunker import chunk_document
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import weaviate
from llm_interaction.llm_handler import LLMHandler
from llm_interaction.prompt_builder import build_rag_prompt, build_system_prompt
from retrieval.retriever import Retriever
# Initialize LLM and Vector Database clients
llm_client = OpenAIClient(base_url=config.get("lm_studio_url"), api_key="lm-studio")
llm_handler = LLMHandler(llm_client=llm_client, chat_model=config.get("chat_model"))
weaviate_client = WeaviateClient()
weaviate_url = config.get("weaviate_url")
weaviate_client.connect(url=weaviate_url, headers={"X-Openai-Api-Key": "lmstudio"})
collection_name = config.get("weaviate_collection_name", "Document")
embedding_model_name = config.get("embedding_model")
azure_endpoint = config.get("azure_di_endpoint")
azure_key = config.get("azure_di_key")

# Initialize Retriever
retriever = Retriever(vector_db_client=weaviate_client, llm_client=llm_client, embedding_model=embedding_model_name)


# Define the desired schema
vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE
        )

vectorizer_config = Configure.Vectorizer.text2vec_openai(
            base_url="http://host.docker.internal:1234",
            model="text-embedding-granite-embedding-278m-multilingual"
        )
properties = [
    Property(name="content", data_type=DataType.TEXT),
    Property(name="token_length", data_type=DataType.INT),
    Property(name="char_length", data_type=DataType.INT),
    Property(name="section_indexes", data_type=DataType.INT_ARRAY),
    Property(name="roles", data_type=DataType.TEXT_ARRAY),
    Property(name="heading", data_type=DataType.TEXT),
    Property(name="page_numbers", data_type=DataType.INT_ARRAY),
    Property(name="filename", data_type=DataType.TEXT),
]

def create_weaviate_collection_if_not_exists(client: WeaviateClient, name: str, vector_index_config: dict, vectorizer_config: dict, properties: list):
    if client.check_collection_exists(name):
        collection = client.get_collection(name)
        print(f"Collection '{name}' already exists.")
        #print(collection)
        return collection
    else:
        print(f"Collection '{name}' does not exist. Creating it...")
        try:
            client.create_collection(
                collection_name=name,
                vector_index_config=vector_index_config,
                vectorizer_config=vectorizer_config,
                properties=properties
            )
            print(f"Collection '{name}' created successfully.")
            return client.get_collection(name)
        except Exception as create_e:
            print(f"Error during collection creation: {create_e}")
            return None# Ensure the collection exists on startup
document_collection = create_weaviate_collection_if_not_exists(
    client=weaviate_client,
    name=collection_name,
    vector_index_config=vector_index_config,
    vectorizer_config=vectorizer_config,
    properties=properties
)
print(type(document_collection))
st.title("Contract QA")

uploaded_file = st.file_uploader("Upload a contract document", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    filename = uploaded_file.name

    if document_collection is not None:
        # Query Weaviate for existing chunks with this filename
        existing_chunks = document_collection.query.fetch_objects(
            filters=Filter.by_property("filename").equal(filename),
            limit=1
        )
        print("Existing chunks: "+str(existing_chunks.objects))
        if existing_chunks.objects:
            st.warning(f"Document '{filename}' appears to have already been processed.")
            if st.button("Process Again Anyway?"):
                st.info("Re-processing document...")
                try:
                    document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
                    chunks = chunk_document(llm_client=llm_client, data=document_data, min_chunk_tokens=config["min_chunk_tokens"], embedding_model=config["embedding_model"], filename=filename)
                    weaviate_client.add_data_objects(collection_name, chunks)
                    st.success(f"Document '{filename}' re-processed and added to the knowledge base.")
                except Exception as e:
                    st.error(f"Error re-processing document: {e}")
        else:
            if st.button("Process Document"):
                st.info("Processing document...")
                try:
                    document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
                    print(document_data)
                    chunks = chunk_document(llm_client=llm_client, data=document_data, min_chunk_tokens=config["min_chunk_tokens"], embedding_model=config["embedding_model"], filename=filename)
                    print(chunks[0])
                    weaviate_client.add_data_objects(collection_name, chunks)
                    st.success(f"Document '{filename}' processed and added to the knowledge base.")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
    else:
        st.error("Failed to connect to or create the Weaviate collection.")

query = st.text_input("Enter your query:", placeholder="e.g. obligations of the service provider")
top_k = st.slider("Top K Results", 1, 10, 5)
alpha = st.slider("Hybrid search alpha", 0.0, 1.0, 0.05)

if query and document_collection:
    with st.spinner("Searching relevant chunks..."):
        results = retriever.retrieve_relevant_chunks(query=query, top_k=top_k, alpha=alpha)
        if results:
            context_chunks = [result.properties["content"] for result in results]
            try:
                augmented_answer = llm_handler.generate_rag_response(query, context_chunks)
                st.subheader("Augmented Answer:")
                st.write(augmented_answer)
            except Exception as e:
                st.error(f"Error generating augmented answer: {e}")

        else:
            st.info("No relevant chunks found.")
elif query and document_collection is None:
    st.error("Cannot perform search. Weaviate collection not initialized.")