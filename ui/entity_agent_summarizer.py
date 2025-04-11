import streamlit as st
from agents.entity_agent import entity_agent, EntityAgentInputSchema, EntityAgentOutputSchema
from core.vector_database.vector_db_client import VectorDBClient
import weaviate
import json


def entity_summarizer_tab(vector_db_client: VectorDBClient):
    st.header("📄 Entity Insight Summarizer")

    # --- FILE INPUT TEXTBOX ---
    with st.sidebar:
        st.subheader("Choose Document")
        selected_filename = st.text_input("Enter the filename (exact match):")

    # --- HELPER: DISPLAY ENTITY INSIGHT ---
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
            prose_parts.append(f"The contract involves the following parties: {', '.join(insight.parties)}.")

        if insight.dates_and_durations:
            prose_parts.append("Key dates and durations mentioned in the contract include:")
            for date in insight.dates_and_durations:
                prose_parts.append(f" - {date}")

        if insight.monetary_values:
            prose_parts.append("The following monetary values are specified:")
            for value in insight.monetary_values:
                prose_parts.append(f" - {value}")

        if insight.obligated_actions:
            prose_parts.append("The contract includes the following obligated actions:")
            for action in insight.obligated_actions:
                prose_parts.append(f" - {action}")

        if not prose_parts:
            prose_parts.append("No specific contract insights were identified.")

        return "\n".join(prose_parts)

    # --- MAIN EXECUTION ---
    if selected_filename:
        st.info(f"Running entity summarizer for: {selected_filename}")

        client = weaviate.connect_to_local()
        collection = client.collections.get("Document")

        # Fetch and filter chunks by filename
        filtered_chunks = [
            item.properties for item in collection.iterator()
            if item.properties.get("filename") == selected_filename
        ]

        # Sort chunks by chunk_number
        sorted_chunks = sorted(filtered_chunks, key=lambda c: c.get("chunk_number", 0))

        insights = []

        for i, chunk in enumerate(sorted_chunks):
            prev_chunks = [c["content"] for c in sorted_chunks[max(0, i - 3):i]]
            prev_insights = [insight.model_dump() for insight in insights[max(0, i - 3):i]]

            entity_input_schema = EntityAgentInputSchema(
                chunk=chunk["content"],
                context_so_far=prev_chunks,
                prior_insights=prev_insights
            )

            result = entity_agent.run(entity_input_schema)

            insights.append(result)
            display_insight(result, i)

        # --- FINAL SUMMARY ---
        if insights:
            final_insight = insights[-1]
            st.subheader("📘 Final Document Insight")
            display_insight(final_insight, len(insights) - 1)

            # JSON Download
            final_json = json.dumps(final_insight.model_dump(), indent=2)
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
