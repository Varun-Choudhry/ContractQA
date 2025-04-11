# rag_app/core/document/chunker.py
import json
from typing import List, Dict, Any, Set
from core.llm.llm_client import LLMClient  # Import our LLMClient
from config.config import config  # Import configuration

# Configuration from config.yaml
EMBED_MODEL = config.get("embedding_model")
MIN_CHUNK_TOKENS = config.get("min_chunk_tokens", 256) # Default to 256 if not in config

def get_embedding(llm_client: LLMClient, text: str, model: str = EMBED_MODEL) -> List[float]:
    """Generates an embedding for the given text using the provided client."""
    return llm_client.embed_text(text, model)

def resolve_reference(ref: str, data: Dict[str, Any]) -> Any:
    """Resolves a reference string to its corresponding object in the JSON data."""
    ref = ref.lstrip("/")
    parts = ref.split("/")
    if len(parts) == 2:
        collection_name = parts[0]
        try:
            index = int(parts[1])
            return data.get(collection_name, [])[index]
        except (ValueError, IndexError):
            return None
    return None

def extract_table_content_html(table: Dict[str, Any], paragraphs: List[Dict[str, Any]]) -> tuple[str, List[int]]:
    """Extracts text content from all cells of a table and formats it as an HTML table."""
    html_table = "<table>\n"
    rows = {}  # Group cells by rowIndex

    # Organize cells by row
    for cell in table.get("cells", []):
        row_index = cell.get("rowIndex", 0)
        if row_index not in rows:
            rows[row_index] = []
        rows[row_index].append(cell)

    # Sort rows by index
    sorted_rows = sorted(rows.items())
    page_numbers: Set[int] = set()

    for row_index, cells_in_row in sorted_rows:
        html_table += "  <tr>\n"
        # Sort cells in the row by columnIndex
        sorted_cells = sorted(cells_in_row, key=lambda c: c.get("columnIndex", 0))
        for cell in sorted_cells:
            element_refs = cell.get("elements", [])
            cell_content = []
            for ref in element_refs:
                paragraph = resolve_reference(ref, {"paragraphs": paragraphs})
                if paragraph and isinstance(paragraph, dict) and "content" in paragraph:
                    cell_content.append(paragraph["content"].strip())
                    for region in paragraph.get("boundingRegions", []):
                        page_numbers.add(region.get("pageNumber"))

            tag = "th" if cell.get("kind") == "columnHeader" else "td"
            html_table += f"    <{tag}>{' '.join(cell_content)}</{tag}>\n"
        html_table += "  </tr>\n"

    html_table += "</table>\n"
    return html_table, list(page_numbers)

def get_page_numbers(element: Dict[str, Any]) -> List[int]:
    """Extracts unique page numbers from the bounding regions of an element."""
    page_numbers: Set[int] = set()
    for region in element.get("boundingRegions", []):
        page_numbers.add(region.get("pageNumber"))
    return list(page_numbers)

def classify_refs(refs: List[str]) -> str:
    """Classifies a list of references based on whether they point to sections."""
    normalized = [ref.lstrip("/") for ref in refs]
    has_sections = any(ref.startswith("sections/") for ref in normalized)
    has_tables = any(ref.startswith("table") for ref in normalized)
    has_non_sections_or_tables = any(not ref.startswith("sections/") and not ref.startswith("table") for ref in normalized)

    if has_sections and not has_non_sections_or_tables and not has_tables:
        return "only_sections"
    elif has_tables and not has_non_sections_or_tables and not has_sections:
        return "only_tables"
    elif (has_sections and has_non_sections_or_tables) or \
            (has_tables and has_non_sections_or_tables) or \
            (has_sections and has_tables):
        return "mixed"
    elif not has_sections and not has_tables and has_non_sections_or_tables:
        return "only_non_sections"
    else:
        return "empty_or_unknown"

def process_section(section: Dict[str, Any], sections: List[Dict[str, Any]], paragraphs: List[Dict[str, Any]], tables: List[Dict[str, Any]], all_section_texts: List[str], current_chunk_texts: List[str], current_roles: List[str], current_token_count: int, current_section_indexes: List[int], current_page_numbers: Set[int], idx: int) -> tuple[List[str], List[str], int, List[int], Set[int]]:
    """Processes a single section to extract text and update the current chunk."""
    elements = section.get("elements", [])
    section_type = classify_refs(elements)
    processed_texts = []
    section_roles = []
    section_page_numbers = get_page_numbers(section)
    current_page_numbers.update(section_page_numbers)

    if section_type == "only_sections":
        print(f"[SKIP] Section {idx} has only section links.")
        return current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers
    elif section_type == "only_tables":
        print(f"[INFO] Section {idx} has only table links. Processing tables directly.")
        for ref in elements:
            table_index_str = ref.lstrip("table")
            try:
                table_index = int(table_index_str)
                if 0 <= table_index < len(tables):
                    html_table, table_pages = extract_table_content_html(tables[table_index], paragraphs)
                    processed_texts.append(html_table)
                    current_page_numbers.update(table_pages)
                else:
                    print(f"[WARN] Invalid table reference: {ref} in Section {idx}")
            except ValueError:
                print(f"[WARN] Invalid table reference format: {ref} in Section {idx}")
    elif section_type in ["only_non_sections", "mixed"]:
        if section_type == "mixed":
            print(f"[MIXED] Section {idx} has mixed content in it.")
        for ref in elements:
            referenced = resolve_reference(ref, {"paragraphs": paragraphs, "sections": sections, "tables": tables})
            if referenced:
                if isinstance(referenced, dict) and "content" in referenced:
                    # It's a paragraph
                    content = referenced.get("content", "").strip()
                    role = referenced.get("role", "")
                    paragraph_pages = get_page_numbers(referenced)
                    current_page_numbers.update(paragraph_pages)
                    if role:
                        section_roles.append(role)
                        formatted = f"[{role.upper()}] {content}"
                    else:
                        formatted = content
                    processed_texts.append(formatted)
                elif isinstance(referenced, dict) and "cells" in referenced:
                    # It's a table
                    html_table, table_pages = extract_table_content_html(referenced, paragraphs)
                    processed_texts.append(html_table)
                    current_page_numbers.update(table_pages)
                else:
                    print(f"[WARN] Invalid reference: {ref} in Section {idx}")
    else:
        print(f"[WARN] Section {idx} has unknown reference structure.")
        return current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers

    section_text = "\n".join(processed_texts)
    section_token_count = len(section_text.split())
    all_section_texts.append(f"[SECTION {idx}]\n{section_text}\n\n")
    current_chunk_texts.append(section_text)
    current_roles.extend(section_roles)
    current_token_count += section_token_count
    current_section_indexes.append(idx)

    return current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers

def create_chunk_object(llm_client: LLMClient, full_text: str, current_section_indexes: List[int], current_roles: List[str], page_numbers: List[int], embedding_model: str, filename:str, chunk_number:int) -> Dict[str, Any]:
    """Creates a data object for a chunk to be inserted into Weaviate."""
    embedding = get_embedding(llm_client, full_text, embedding_model)
    token_count = len(full_text.split())
    char_count = len(full_text)

    heading = None
    lines = full_text.split('\n', 1)  # Split only the first line
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("[TITLE]") or first_line.startswith("[SECTIONHEADING]"):
            heading = first_line
            # Optionally, remove the heading from the content if you don't want it duplicated
            full_text = lines[1].strip() if len(lines) > 1 else ""

    return {
        "content": full_text,
        "token_length": token_count,
        "char_length": char_count,
        "section_indexes": current_section_indexes.copy(),
        "roles": list(set(current_roles)),
        "heading": heading,  # Add the extracted heading
        "page_numbers": sorted(list(set(page_numbers))), # Ensure unique and sorted page numbers
        "chunk_number" : chunk_number,
        "filename": filename,

        "_additional": {"vector": embedding}
    }

def chunk_document(llm_client: LLMClient, data: Dict[str, Any], min_chunk_tokens: int, embedding_model: str, filename:str) -> List[Dict[str, Any]]:
    """Processes the JSON data and chunks it."""
    sections = data.get("sections", [])
    paragraphs = data.get("paragraphs", [])
    tables = data.get("tables", [])
    data_objects = []
    all_chunks = []
    all_section_texts = []
    current_chunk_texts = []
    current_roles = []
    current_token_count = 0
    current_section_indexes = []
    current_page_numbers: Set[int] = set()
    chunk_number = 1

    # Collect all section texts
    for idx, section in enumerate(sections):
        elements = section.get("elements", [])
        section_type = classify_refs(elements)
        processed_texts = []
        if section_type == "only_sections":
            print(f"[SKIP] Section {idx} has only section links.")
        elif section_type == "only_tables":
            for ref in elements:
                table_index_str = ref.lstrip("table")
                try:
                    table_index = int(table_index_str)
                    if 0 <= table_index < len(tables):
                        html_table, _ = extract_table_content_html(tables[table_index], paragraphs)
                        processed_texts.append(html_table)
                except ValueError:
                    pass
        elif section_type in ["only_non_sections", "mixed"]:
            for ref in elements:
                referenced = resolve_reference(ref, {"paragraphs": paragraphs, "sections": sections, "tables": tables})
                if referenced and isinstance(referenced, dict) and "content" in referenced:
                    processed_texts.append(referenced["content"].strip())
                elif referenced and isinstance(referenced, dict) and "cells" in referenced:
                    html_table, _ = extract_table_content_html(referenced, paragraphs)
                    processed_texts.append(html_table)

        section_text = "\n".join(processed_texts)
        all_section_texts.append(f"[SECTION {idx}]\n{section_text}\n\n")

    with open("sections.txt", "w", encoding="utf-8") as f_sections:
        f_sections.writelines(all_section_texts)
        print("✅ All sections written to sections.txt")

    # Chunk the document
    for idx, section in enumerate(sections):
        current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers = process_section(
            section, sections, paragraphs, tables, all_section_texts, current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers, idx
        )

        if current_token_count >= min_chunk_tokens:
            full_text = "\n".join(current_chunk_texts)
            data_object = create_chunk_object(llm_client, full_text, current_section_indexes, current_roles, list(current_page_numbers), embedding_model, filename, chunk_number)
            data_objects.append(data_object)
            all_chunks.append(f"[CHUNK composed of sections {current_section_indexes}]\n{full_text}\n\n")
            chunk_number += 1
            current_chunk_texts = []
            current_roles = []
            current_token_count = 0
            current_section_indexes = []
            current_page_numbers = set()

    # Final chunk
    if current_chunk_texts:
        full_text = "\n".join(current_chunk_texts)
        data_object = create_chunk_object(llm_client, full_text, current_section_indexes, current_roles, list(current_page_numbers), embedding_model, filename, chunk_number)
        data_objects.append(data_object)
        all_chunks.append(f"[CHUNK composed of sections {current_section_indexes}]\n{full_text}\n\n")

    print(f"Total number of chunks (sections): {len(data_objects)}")

    with open("chunks.txt", "w", encoding="utf-8") as f_chunks:
        f_chunks.writelines(all_chunks)
        print("✅ Chunks written to chunks.txt")

    return data_objects