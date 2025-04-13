import streamlit as st
from agents.entity_agent import entity_agent, EntityAgentInputSchema, EntityAgentOutputSchema
from core.vector_database.vector_db_client import VectorDBClient
from core.mongodb.mongo_client import MongoDBClient
import weaviate
import json

DOCUMENT_STATUS = ["uploaded", "processed", "summarized"]
def entity_summarizer_tab(vector_db_client: VectorDBClient, mongo_client: MongoDBClient):
    st.header("📄 Entity Insight Summarizer")

    available_docs_with_status = mongo_client.get_all_documents_with_status()
    if not available_docs_with_status:
        st.info("No documents have been uploaded yet.")
        return

    def format_doc_display(doc):
        return f"{doc['filename']} ({doc['status'].capitalize()})"

    selected_doc_data = st.selectbox("Choose a Document:", available_docs_with_status, format_func=format_doc_display)
    selected_filename = selected_doc_data['filename']
    selected_status = selected_doc_data['status']

    def display_insight(insight: EntityAgentOutputSchema, index: int):
        with st.expander(f"Chunk {index} - Insight", expanded=True):
            st.markdown("### Contract Parties")
            st.write(insight.parties)
            st.markdown("### Dates & Durations")
            st.write(insight.dates_and_durations)
            st.markdown("### Monetary Values")
            st.write(insight.monetary_values)
            st.markdown("### Obligated Actions")
            st.write(insight.obligated_actions)

    def generate_prose_summary(insight: EntityAgentOutputSchema) -> str:
        prose_parts = []
        if insight.parties:
            party_names = [party.get('name') for party in insight.parties if isinstance(party, dict) and party.get('name')]
            prose_parts.append(f"The contract involves the following parties: {', '.join(party_names)}.") if party_names else prose_parts.append("The contract identifies parties, but their names could not be extracted.")
        if insight.dates_and_durations:
            dates_info = [f"{item.get('value')} ({item.get('context')})" for item in insight.dates_and_durations if item.get('value')]
            prose_parts.append(f"Key dates and durations mentioned in the contract include: {', '.join(dates_info)}.") if dates_info else prose_parts.append("Key dates and durations were identified but could not be formatted.")
        if insight.monetary_values:
            monetary_info = [f"{item.get('value')} ({item.get('context')})" for item in insight.monetary_values if item.get('value')]
            prose_parts.append(f"The following monetary values are specified: {', '.join(monetary_info)}.") if monetary_info else prose_parts.append("Monetary values were identified but could not be formatted.")
        if insight.obligated_actions:
            valid_actions = [action for action in insight.obligated_actions if isinstance(action, str) and action.strip()]
            prose_parts.append(f"The contract includes the following obligated actions: {', '.join(valid_actions)}.") if valid_actions else prose_parts.append("Obligated actions were identified but could not be formatted.")
        return "\n".join(prose_parts) if prose_parts else "No specific contract insights were identified."

    def run_summarization():
        st.info(f"Running entity summarizer for: {selected_filename}")
        mongo_client.update_document_status(selected_filename, "processing")

        client = weaviate.connect_to_local()
        collection = client.collections.get("Document")
        filtered_chunks = [item.properties for item in collection.iterator() if item.properties.get("filename") == selected_filename]
        sorted_chunks = sorted(filtered_chunks, key=lambda c: c.get("chunk_number", 0))

        insights = []
        for i, chunk in enumerate(sorted_chunks):
            prev_chunks = [c["content"] for c in sorted_chunks[max(0, i - 1):i]]
            prev_insights = [insight.model_dump() for insight in insights[max(0, i - 3):i]]
            entity_input_schema = EntityAgentInputSchema(chunk=chunk["content"], context_so_far=prev_chunks, insights=prev_insights)
            result = entity_agent.run(entity_input_schema)
            insights.append(result)

        if insights:
            final_insight = insights[-1]
            st.subheader("📘 Final Document Insight")
            display_insight(final_insight, len(insights) - 1)
            insight_json = final_insight.model_dump()
            mongo_client.update_document_insight(selected_filename, insight_json)
            mongo_client.update_document_status(selected_filename, "summarized")
            st.success(f"Entity insight for '{selected_filename}' has been generated and stored.")

            # JSON Download
            final_json = json.dumps(insight_json, indent=2)
            st.download_button(
                label="📥 Download Final Insight as JSON",
                data=final_json,
                file_name=f"{selected_filename}_insight.json",
                mime="application/json"
            )

            # Prose Summary
            st.subheader("📝 Final Prose Summary")
            prose_summary = generate_prose_summary(final_insight)
            st.text(prose_summary)
        else:
            mongo_client.update_document_status(selected_filename, "processed")
            st.warning("Could not generate entity insights for this document.")

    # --- MAIN EXECUTION ---
    if selected_filename:
        rerun_clicked = False

        if selected_status == "summarized":
            st.info(f"Displaying previously generated insight for: {selected_filename}")

            existing_insight = mongo_client.get_document_insight(selected_filename)
            st.write("DEBUG: Retrieved MongoDB document", existing_insight)

            if existing_insight and 'json_insight' in existing_insight:
                st.subheader("💾 Stored JSON Insight")
                st.json(existing_insight['json_insight'], expanded=True)

                # Re-summarize button
                rerun_clicked = st.button("🔁 Re-run summarization")

                if rerun_clicked:
                    run_summarization()
            else:
                st.warning("No stored JSON insight found for this document.")
        else:
            run_summarization()
