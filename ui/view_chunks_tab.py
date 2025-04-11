import streamlit as st
from weaviate.classes.query import Filter

def view_chunks_tab(document_collection):
    st.title("📦 View Chunks in Weaviate")

    all_chunks = document_collection.query.fetch_objects(limit=100)  # Adjust limit as needed

    if not all_chunks.objects:
        st.info("No chunks found in Weaviate.")
        return

    for obj in all_chunks.objects:
        st.markdown("### 🔹 Chunk")
        st.write({
            "content": obj.properties.get("content", "")[:500],  # Show preview of content
            "token_length": obj.properties.get("token_length"),
            "char_length": obj.properties.get("char_length"),
            "section_indexes": obj.properties.get("section_indexes"),
            "roles": obj.properties.get("roles"),
            "heading": obj.properties.get("heading"),
            "page_numbers": obj.properties.get("page_numbers"),
            "filename": obj.properties.get("filename"),
            "chunk_number": obj.properties.get("chunk_number")
        })
