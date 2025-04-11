import streamlit as st
from config.config import config
from core.llm.openai_client import OpenAIClient
from core.vector_database.weaviate_client import WeaviateClient
from core.document.document_loader import load_document_from_upload
from core.document.chunker import chunk_document
from weaviate.classes.query import Filter
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from retrieval.retriever import Retriever
from agents.decompose_query_agent import decompose_query_agent,DecomposeInputSchema
from agents.metadata_matcher_agent import metadata_matcher_agent,MetadataMatcherInputSchema
from agents.final_answer_agent import final_answer_agent,FinalAnswerInputSchema
from tools.hybrid_search_tool import HybridSearchTool, HybridSearchToolInputSchema
from tools.targeted_search_tool import TargetedSearchTool

# ---- Init Clients ----
llm_client = OpenAIClient(base_url=config["lm_studio_url"], api_key="lm-studio")
weaviate_client = WeaviateClient()
weaviate_client.connect(config["weaviate_url"], headers={"X-Openai-Api-Key": "lmstudio"})

collection_name = config.get("weaviate_collection_name", "Document")
embedding_model_name = config.get("embedding_model")
azure_endpoint = config.get("azure_di_endpoint")
azure_key = config.get("azure_di_key")

retriever = Retriever(vector_db_client=weaviate_client, llm_client=llm_client, embedding_model=embedding_model_name)

# ---- Init Tools ----
hybrid_tool = HybridSearchTool(retriever)
targeted_tool = TargetedSearchTool(retriever)

# ---- Ensure Collection Exists ----
def ensure_collection(client):
    vector_index_config = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
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
        Property(name="chunk_number", data_type=DataType.INT)
    ]
    if not client.check_collection_exists(collection_name):
        client.create_collection(collection_name, vector_index_config, vectorizer_config, properties)
    return client.get_collection(collection_name)

document_collection = ensure_collection(weaviate_client)

# ---- Streamlit UI ----
st.title("Contract QA with Agents")

uploaded_file = st.file_uploader("Upload a contract document", type=["pdf", "txt", "docx"])

if uploaded_file:
    filename = uploaded_file.name
    st.session_state["filename"] = filename

    existing_chunks = document_collection.query.fetch_objects(
        filters=Filter.by_property("filename").equal(filename), limit=1
    )

    if existing_chunks.objects:
        st.warning(f"Document '{filename}' already exists.")
        if st.button("Reprocess Document"):
            document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
            chunks = chunk_document(llm_client, document_data, config["min_chunk_tokens"], embedding_model_name, filename)
            weaviate_client.add_data_objects(collection_name, chunks)
            st.success("Document re-processed successfully.")
    else:
        if st.button("Process Document"):
            document_data = load_document_from_upload(azure_endpoint, azure_key, uploaded_file)
            chunks = chunk_document(llm_client, document_data, config["min_chunk_tokens"], embedding_model_name, filename)
            weaviate_client.add_data_objects(collection_name, chunks)
            st.success("Document processed successfully.")

# ---- Query Interface ----
query = st.text_input("Ask a question about the contract:", placeholder="e.g. What are the key obligations?")

if query and document_collection:
    with st.spinner("Thinking..."):
        decompose_input_schema = DecomposeInputSchema(query=query)
        decomp = decompose_query_agent.run(decompose_input_schema)
        sub_queries = decomp.subqueries or [query]
        all_results = []

        for q in sub_queries: #Commenting Metadata analyzer agent because of inconsistent behaviour until further analysis
            #metadata_input_schema = MetadataMatcherInputSchema(query=q)
            #match = metadata_matcher_agent.run(metadata_input_schema)
            #if match.matches_metadata:
            #   results = targeted_tool.run({
            #       "query": q,
            #        "metadata_key": match.matched_property,
            #       "metadata_value": match.value
            #    })
            #else:
            #    hybrid_input = HybridSearchToolInputSchema(query=q, top_k=5)
            #    results = hybrid_tool.run(hybrid_input)
            hybrid_input = HybridSearchToolInputSchema(query=q, top_k=5)
            results = hybrid_tool.run(hybrid_input)
            all_results.extend(results.results)

        # ✅ Deduplicate by chunk 'text'
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


from ui.view_chunks_tab import view_chunks_tab
from ui.entity_agent_summarizer import entity_summarizer_tab  # You'll create this next

tab = st.sidebar.selectbox("Choose a tab", ["Main QA", "View Chunks", "Entity Summarizer"])

if tab == "View Chunks":
    view_chunks_tab(document_collection)
elif tab == "Entity Summarizer":
    entity_summarizer_tab(vector_db_client=weaviate_client)
    