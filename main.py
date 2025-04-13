import streamlit as st
from config.config import config
from core.llm.azure_openai_client import AzureOpenAIClient
from core.vector_database.weaviate_client import WeaviateClient
from core.mongodb.mongo_client import MongoDBClient
from core.document.document_loader import load_document_from_upload
from core.document.chunker import chunk_document
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from retrieval.retriever import Retriever
from agents.decompose_query_agent import decompose_query_agent, DecomposeInputSchema
from agents.final_answer_agent import final_answer_agent, FinalAnswerInputSchema
from tools.hybrid_search_tool import HybridSearchTool, HybridSearchToolInputSchema
from tools.targeted_search_tool import TargetedSearchTool
from ui.view_chunks_tab import view_chunks_tab
from ui.entity_agent_summarizer import entity_summarizer_tab


llm_client = AzureOpenAIClient(
    chat_version=config["azure_openai_chat_api_version"],
    chat_endpoint=config["azure_openai_chat_endpoint"],
    chat_key=config["azure_openai_chat_key"],
    embedding_version=config["azure_openai_embedding_api_version"],
    embedding_endpoint=config["azure_openai_embedding_endpoint"],
    embedding_key=config["azure_openai_embedding_key"]
)
weaviate_client = WeaviateClient()
weaviate_client.connect(config["weaviate_url"], headers={"X-Azure-Api-Key": config["azure_openai_embedding_key"]})
mongo_client = MongoDBClient()

collection_name = config.get("weaviate_collection_name", "Document")
embedding_model_name = config.get("azure_openai_embedding_model")
azure_endpoint = config.get("azure_di_endpoint")
azure_key = config.get("azure_di_key")

retriever = Retriever(vector_db_client=weaviate_client, llm_client=llm_client, embedding_model=embedding_model_name)
hybrid_tool = HybridSearchTool(retriever)
targeted_tool = TargetedSearchTool(retriever)

# ---- Ensure Collection Exists ----
def ensure_collection(client):
    vector_index_config = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
    vectorizer_config = Configure.Vectorizer.text2vec_azure_openai(
        model=config["azure_openai_embedding_model"],
        resource_name="varun-m32lypz9-eastus",
        deployment_id="text-embedding-3-large"
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
        Property(name="chunk_number", data_type=DataType.INT)
    ]
    if not client.check_collection_exists(collection_name):
        client.create_collection(collection_name, vector_index_config, vectorizer_config, properties)
    return client.get_collection(collection_name)

document_collection = ensure_collection(weaviate_client)


st.title("Contract QA with Agents")

tab = st.sidebar.selectbox("Choose a tab", ["Upload Document", "Main QA", "View Chunks", "Entity Summarizer"])

if tab == "Upload Document":
    uploaded_file = st.file_uploader("Upload a contract document", type=["pdf", "txt", "docx"])

    if uploaded_file:
        filename = uploaded_file.name

        # Check MongoDB for existing document data
        doc_info = mongo_client.get_document_by_filename(filename)
        if doc_info:
            st.warning(f"Document '{filename}' already exists (Status: {doc_info.get('status', 'Unknown').capitalize()}).")
            if doc_info["status"] == "uploaded" or doc_info["status"] == "processed":
                if st.button("Reprocess Document"):
                    # If already uploaded but not processed, proceed with chunking and uploading to Weaviate
                    document_data = doc_info.get("data", {})
                    if not document_data:  # If data is empty, process the document again
                        st.warning(f"Document data not found. Re-processing document...")
                        document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
                        mongo_client.update_document_data(filename, document_data)  # Update MongoDB with new data
                    
                    # Chunk document after loading from MongoDB
                    chunks = chunk_document(
                        llm_client=llm_client,
                        data=document_data,
                        min_chunk_tokens=config["min_chunk_tokens"],
                        embedding_model=embedding_model_name,
                        filename=filename
                    )

                    # Add chunks to Weaviate
                    weaviate_client.add_data_objects(collection_name, chunks)
                    mongo_client.update_document_status(filename, "processed")  # Update status to processed
                    st.success("Document re-processed successfully.")
            else:
                st.warning(f"Document is already processed and uploaded.")
        else:
            if st.button("Process Document"):
                # If document is not processed before, process and save data
                document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
                # Store the Azure DI output (document data) in MongoDB
                mongo_client.insert_document({"filename": filename, "data": document_data, "status": "uploaded"})
                
                # Chunk document after loading from Azure DI output
                chunks = chunk_document(
                    llm_client=llm_client,
                    data=document_data,
                    min_chunk_tokens=config["min_chunk_tokens"],
                    embedding_model=embedding_model_name,
                    filename=filename
                )

                # Add chunks to Weaviate
                weaviate_client.add_data_objects(collection_name, chunks)
                mongo_client.update_document_status(filename, "processed")  # Update status to processed
                st.success("Document uploaded, data saved to MongoDB, and chunked successfully.")

# ---- Main QA Tab ----
elif tab == "Main QA":
    available_docs = mongo_client.get_all_filenames()
    selected_file = st.selectbox("Select a document to query", available_docs)

    query = st.text_input("Ask a question about the contract:", placeholder="e.g. What are the key obligations?")

    if query and selected_file:
        with st.spinner("Thinking..."):
            decompose_input_schema = DecomposeInputSchema(query=query)
            decomp = decompose_query_agent.run(decompose_input_schema)
            sub_queries = decomp.subqueries or [query]
            all_results = []
            for q in sub_queries:
                hybrid_input = HybridSearchToolInputSchema(query=q, top_k=5, filename=selected_file)
                results = hybrid_tool.run(hybrid_input) 
                all_results.extend(results.results)

            seen = set()
            deduped_chunks = []
            for chunk in all_results:
                if chunk not in seen:
                    seen.add(chunk)
                    deduped_chunks.append(chunk)

            final_input_schema = FinalAnswerInputSchema(query=query, retrieved_chunks=deduped_chunks)
            answer = final_answer_agent.run(final_input_schema)
            st.subheader("Answer")
            st.write(answer.answer)

# ---- Other Tabs ----
elif tab == "View Chunks":
    view_chunks_tab(document_collection)

elif tab == "Entity Summarizer":
    entity_summarizer_tab(vector_db_client=weaviate_client, mongo_client=mongo_client)
