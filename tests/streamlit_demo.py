import streamlit as st
import io
from orchestrator import Orchestrator

pipeline = [
    "convert_document_action:semantic",
    "chunker_action:fixed",
    "embedder_action:azure",
    "vector_upload_action:weaviatelocal"
]

pipeline2 = [
    "embedder_action:azure",
    "hybrid_retrieval_action:weaviatelocal",
    "final_answer_action:azure"
]


st.title("RAG Pipeline Demo")

tab1, tab2 = st.tabs(["Upload & Chunk", "Ask a Question"])

with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a document", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("Document uploaded!")

        if st.button("Run Chunking Pipeline"):
            doc_stream = io.BytesIO(uploaded_file.read())

            first_input = {
                "document": doc_stream
            }

            orchestrator = Orchestrator(pipeline=pipeline, first_input=first_input)
            output = orchestrator.run()


with tab2:
    st.header("Ask a Question")
    question = st.text_area("Enter your question")

    if st.button("Run QA Pipeline"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            first_input = {
                "chunks": [question]
            }

            orchestrator = Orchestrator(pipeline=pipeline2, first_input=first_input)
            output = orchestrator.run()
