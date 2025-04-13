# rag_app/core/document/chunker.py
import json
from typing import List, Dict, Any, Set
from core.llm.llm_client import LLMClient  
from config.config import config  
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Node:
    def __init__(self, value, details=None):
        self.value = value
        self.details = details if details else {}
        self.children = []

def build_tree_from_analyze_result(analyze_result):
    root_node = Node("/analyzeResult")
    paragraph_nodes = {}
    table_nodes = {}

    if "paragraphs" in analyze_result and isinstance(analyze_result["paragraphs"], list):
        paragraph_root = Node("/paragraphs")
        root_node.children.append(paragraph_root)
        for i, paragraph_data in enumerate(analyze_result["paragraphs"]):
            paragraph_id = f"/paragraphs/{i}"
            node = Node(paragraph_id, details=paragraph_data)
            paragraph_root.children.append(node)
            paragraph_nodes[paragraph_id] = node

    if "tables" in analyze_result and isinstance(analyze_result["tables"], list):
        table_root = Node("/tables")
        root_node.children.append(table_root)
        for i, table_data in enumerate(analyze_result["tables"]):
            table_id = f"/tables/{i}"
            table_node = Node(table_id, details=table_data)
            table_root.children.append(table_node)
            table_nodes[table_id] = table_node
            if "cells" in table_data and isinstance(table_data["cells"], list):
                cells_root = Node(f"{table_id}/cells")
                table_node.children.append(cells_root)
                for cell_data in table_data["cells"]:
                    cell_id = f"{table_id}/cells/row_{cell_data.get('rowIndex')}_col_{cell_data.get('columnIndex')}"
                    cell_node = Node(cell_id, details=cell_data)
                    cells_root.children.append(cell_node)
                    if "elements" in cell_data and isinstance(cell_data["elements"], list):
                        for element_id in cell_data["elements"]:
                            if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                                cell_node.children.append(paragraph_nodes[element_id])
            if "caption" in table_data and isinstance(table_data["caption"], dict) and "elements" in table_data["caption"]:
                for element_id in table_data["caption"]["elements"]:
                    if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                        table_node.children.append(paragraph_nodes[element_id])

    if "sections" in analyze_result and isinstance(analyze_result["sections"], list):
        section_root = Node("/sections")
        root_node.children.append(section_root)
        for i, section_data in enumerate(analyze_result["sections"]):
            section_id = f"/sections/{i}"
            section_node = Node(section_id, details=section_data)
            section_root.children.append(section_node)
            if "elements" in section_data and isinstance(section_data["elements"], list):
                for element_id in section_data["elements"]:
                    if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                        section_node.children.append(paragraph_nodes[element_id])
                    elif element_id.startswith("/tables/") and element_id in table_nodes:
                        section_node.children.append(table_nodes[element_id])

    return [root_node]

def build_tree_from_direct_output(data):
    root_node = Node("/root")
    paragraph_nodes = {}
    table_nodes = {}

    logging.info(f"Input data to build_tree_from_direct_output: {data}")

    if "paragraphs" in data and isinstance(data["paragraphs"], list):
        paragraph_root = Node("/paragraphs")
        root_node.children.append(paragraph_root)
        for i, paragraph_data in enumerate(data["paragraphs"]):
            paragraph_id = f"/paragraphs/{i}"
            node = Node(paragraph_id, details=paragraph_data)
            paragraph_root.children.append(node)
            paragraph_nodes[paragraph_id] = node
            logging.info(f"Paragraph node created: {paragraph_id} with details: {paragraph_data}")

    if "tables" in data and isinstance(data["tables"], list):
        table_root = Node("/tables")
        root_node.children.append(table_root)
        for i, table_data in enumerate(data["tables"]):
            table_id = f"/tables/{i}"
            table_node = Node(table_id, details=table_data)
            table_root.children.append(table_node)
            table_nodes[table_id] = table_node
            logging.info(f"Table node created: {table_id} with details: {table_data}")
            if "cells" in table_data and isinstance(table_data["cells"], list):
                cells_root = Node(f"{table_id}/cells")
                table_node.children.append(cells_root)
                for cell_data in table_data["cells"]:
                    cell_id = f"{table_id}/cells/row_{cell_data.get('rowIndex')}_col_{cell_data.get('columnIndex')}"
                    cell_node = Node(cell_id, details=cell_data)
                    cells_root.children.append(cell_node)
                    logging.info(f"Cell node created: {cell_id} with details: {cell_data}")
                    if "elements" in cell_data and isinstance(cell_data["elements"], list):
                        for element_id in cell_data["elements"]:
                            if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                                cell_node.children.append(paragraph_nodes[element_id])
                                logging.info(f"Cell node appended with paragraph: {element_id}")
                            elif element_id.startswith("/tables/") and element_id in table_nodes:
                                cell_node.children.append(table_nodes[element_id])
                                logging.info(f"Cell node appended with table: {element_id}")
            if "caption" in table_data and isinstance(table_data["caption"], dict) and "elements" in table_data["caption"]:
                for element_id in table_data["caption"]["elements"]:
                    if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                        table_node.children.append(paragraph_nodes[element_id])
                        logging.info(f"Table node appended with paragraph from caption: {element_id}")

    if "sections" in data and isinstance(data["sections"], list):
        section_root = Node("/sections")
        root_node.children.append(section_root)
        for i, section_data in enumerate(data["sections"]):
            section_id = f"/sections/{i}"
            section_node = Node(section_id, details=section_data)
            section_root.children.append(section_node)
            logging.info(f"Section node created: {section_id} with details: {section_data}")
            if "elements" in section_data and isinstance(section_data["elements"], list):
                for element_id in section_data["elements"]:
                    if element_id.startswith("/paragraphs/") and element_id in paragraph_nodes:
                        section_node.children.append(paragraph_nodes[element_id])
                        logging.info(f"Section node appended with paragraph: {element_id}")
                    elif element_id.startswith("/tables/") and element_id in table_nodes:
                        section_node.children.append(table_nodes[element_id])
                        logging.info(f"Section node appended with table: {element_id}")

    return [root_node]

def get_paragraph_text(node):
    content = node.details.get("content", "")
    role = node.details.get("role")
    if role == "title":
        return f"<h1>{content}</h1>"
    elif role in ["heading1"]:
        return f"<h2>{content}</h2>"
    elif role in ["heading2"]:
        return f"<h3>{content}</h3>"
    elif role in ["heading3"]:
        return f"<h4>{content}</h4>"
    elif role in ["heading4"]:
        return f"<h5>{content}</h5>"
    elif role in ["heading5"]:
        return f"<h6>{content}</h6>"
    elif role == "pageHeader":
        return f"<h4>{content}</h4>"
    return content

def get_html_table_content_from_node(node):
    table_details = node.details
    if not table_details:
        return ""

    rows_data = {}
    has_content = False

    for cell_node in node.children:
        if cell_node.value.startswith(node.value + "/cells/"):
            cell_details = cell_node.details
            if cell_details and "rowIndex" in cell_details and "columnIndex" in cell_details:
                has_content = True
                row_index = cell_details["rowIndex"]
                col_index = cell_details["columnIndex"]
                content = cell_details.get("content", "")
                tag = "td"
                if cell_details.get("kind") == "columnHeader" or cell_details.get("kind") == "rowHeader":
                    tag = "th"
                row_key = f"row_{row_index}"
                if row_key not in rows_data:
                    rows_data[row_key] = []
                rows_data[row_key].append({
                    "columnIndex": col_index,
                    "content": f"<{tag}>{content}</{tag}>",
                    "rowSpan": cell_details.get("rowSpan", 1),
                    "colSpan": cell_details.get("columnSpan", 1)
                })

    if not has_content and not table_details.get("caption"):
        return ""

    html = "<table border='1'>"
    sorted_rows = sorted(rows_data.items(), key=lambda item: int(item[0].split('_')[1]))

    for _, cell_data_list in sorted_rows:
        html += "<tr>"
        sorted_cells = sorted(cell_data_list, key=lambda cell: cell["columnIndex"])
        for cell in sorted_cells:
            rowspan_attr = f' rowspan="{cell["rowSpan"]}"' if cell["rowSpan"] > 1 else ''
            colspan_attr = f' colspan="{cell["colSpan"]}"' if cell["colSpan"] > 1 else ''
            html += f'{cell["content"]}{rowspan_attr}{colspan_attr}'
        html += "</tr>"

    caption = table_details.get("caption", {}).get("content")
    if caption:
        html = f"<caption>{caption}</caption>" + html

    html += "</table>"
    return html

def chunk_by_hierarchy(nodes, current_chunk="", chunks=[], min_words=256, max_words=2048):
    for node in nodes:
        if node.value.startswith("/paragraphs/"):
            paragraph_text = get_paragraph_text(node)
            words = re.findall(r'\b\w+\b', current_chunk + " " + paragraph_text)
            if len(words) <= max_words:
                current_chunk += paragraph_text + "\n\n"
                logging.info(f"Added paragraph to current chunk. current_chunk: {current_chunk[:50]}...")
            else:
                if len(re.findall(r'\b\w+\b', current_chunk)) >= min_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph_text + "\n\n"
                    logging.info(f"Chunk created and paragraph added. chunks count: {len(chunks)}, current_chunk: {current_chunk[:50]}...")
                else:
                    current_chunk += paragraph_text + "\n\n"
                    logging.info(f"Paragraph added to current chunk (below min words). current_chunk: {current_chunk[:50]}...")
        elif node.value.startswith("/tables/"):
            table_html = get_html_table_content_from_node(node)
            if table_html:
                words = re.findall(r'\b\w+\b', current_chunk + " " + table_html)
                if len(words) <= max_words:
                    current_chunk += table_html + "\n\n"
                    logging.info(f"Added table to current chunk. current_chunk: {current_chunk[:50]}...")
                else:
                    if len(re.findall(r'\b\w+\b', current_chunk)) >= min_words:
                        chunks.append(current_chunk.strip())
                        chunks.append(table_html + "\n\n")
                        current_chunk = ""
                        logging.info(f"Chunk created and table added. chunks count: {len(chunks)}, current_chunk: {current_chunk}")
                    else:
                        chunks.append(table_html + "\n\n")
                        current_chunk = ""
                        logging.info(f"Table added as a separate chunk (below min words). chunks count: {len(chunks)}, current_chunk: {current_chunk}")
        elif node.value.startswith("/sections/"):
            current_chunk, chunks = chunk_by_hierarchy(node.children, current_chunk, chunks, min_words, max_words)
            logging.info(f"Processed section. current_chunk: {current_chunk[:50]}..., chunks count: {len(chunks)}")

    return current_chunk, chunks

def process_tree_and_chunk(nodes, min_words=256, max_words=2048):
    all_chunks = []
    current_chunk = ""
    for root in nodes:
        current_chunk, chunks = chunk_by_hierarchy(root.children, current_chunk, [], min_words, max_words)
        logging.info(f"Current chunk after processing children: {current_chunk[:50]}...")
        logging.info(f"Chunks after processing children: {len(chunks)}")
        all_chunks.extend(chunks)
        if current_chunk.strip():
            all_chunks.append(current_chunk.strip())
            current_chunk = ""
    logging.info(f"All chunks before final processing: {len(all_chunks)}")

    final_chunks = []
    temp_chunk = ""
    for chunk in all_chunks:
        words = re.findall(r'\b\w+\b', temp_chunk + " " + chunk)
        if len(words) <= max_words:
            temp_chunk += " " + chunk
            logging.info(f"Added chunk to temp_chunk: {temp_chunk[:50]}...")
        else:
            if len(re.findall(r'\b\w+\b', temp_chunk)) >= min_words:
                final_chunks.append(temp_chunk.strip())
                temp_chunk = chunk
                logging.info(f"Chunk added to final_chunks, temp_chunk reset: {temp_chunk[:50]}..., final_chunks count: {len(final_chunks)}")
            else:
                temp_chunk += " " + chunk
                logging.info(f"Chunk added to temp_chunk (below min words): {temp_chunk[:50]}...")

    if temp_chunk.strip():
        final_chunks.append(temp_chunk.strip())
        logging.info(f"Final temp_chunk added: {temp_chunk[:50]}..., final_chunks count: {len(final_chunks)}")
    logging.info(f"Final chunks count: {len(final_chunks)}")
    return [chunk for chunk in final_chunks if len(re.findall(r'\b\w+\b', chunk)) >= min_words]

# --- Existing functions ---
EMBED_MODEL = config.get("embedding_model")
MIN_CHUNK_TOKENS = config.get("min_chunk_tokens", 256)

def get_embedding(llm_client: LLMClient, text: str, model: str = EMBED_MODEL) -> List[float]:
    return llm_client.embed_text(text, model)

def resolve_reference(ref: str, data: Dict[str, Any]) -> Any:
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
    html_table = "<table>\n"
    rows = {}

    for cell in table.get("cells", []):
        row_index = cell.get("rowIndex", 0)
        if row_index not in rows:
            rows[row_index] = []
        rows[row_index].append(cell)

    sorted_rows = sorted(rows.items())
    page_numbers: Set[int] = set()

    for row_index, cells_in_row in sorted_rows:
        html_table += "  <tr>\n"
        sorted_cells = sorted(cells_in_row, key=lambda cell: cell["columnIndex"])
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
    page_numbers: Set[int] = set()
    for region in element.get("boundingRegions", []):
        page_numbers.add(region.get("pageNumber"))
    return list(page_numbers)

def classify_refs(refs: List[str]) -> str:
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
    elements = section.get("elements", [])
    section_type = classify_refs(elements)
    processed_texts = []
    section_roles = []
    section_page_numbers = get_page_numbers(section)
    current_page_numbers.update(section_page_numbers)

    logging.info(f"Processing section: {idx} with type: {section_type}")

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
                    logging.info(f"Processed table {table_index} from section {idx}")
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
                    logging.info(f"Processed paragraph from section {idx}: {formatted[:50]}...")
                elif isinstance(referenced, dict) and "cells" in referenced:
                    # It's a table
                    html_table, table_pages = extract_table_content_html(referenced, paragraphs)
                    processed_texts.append(html_table)
                    current_page_numbers.update(table_pages)
                    logging.info(f"Processed table from section {idx}: {html_table[:50]}...")
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
    logging.info(f"Section {idx} processed. current_chunk_texts count: {len(current_chunk_texts)}, current_roles: {current_roles}, current_token_count: {current_token_count}, current_section_indexes: {current_section_indexes}")

    return current_chunk_texts, current_roles, current_token_count, current_section_indexes, current_page_numbers

def create_chunk_object(llm_client: LLMClient, full_text: str, current_section_indexes: List[int], current_roles: List[str], page_numbers: List[int], embedding_model: str, filename:str, chunk_number:int) -> Dict[str, Any]:
    embedding = get_embedding(llm_client, full_text, embedding_model)
    token_count = len(full_text.split())
    char_count = len(full_text)

    heading = None
    lines = full_text.split('\n', 1)
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("[TITLE]") or first_line.startswith("[SECTIONHEADING]"):
            heading = first_line
            full_text = lines[1].strip() if len(lines) > 1 else ""

    chunk_object = {
        "content": full_text,
        "token_length": token_count,
        "char_length": char_count,
        "section_indexes": current_section_indexes.copy(),
        "roles": list(set(current_roles)),
        "heading": heading,
        "page_numbers": sorted(list(set(page_numbers))),
        "chunk_number" : chunk_number,
        "filename": filename,
        "_additional": {"vector": embedding}
    }
    return chunk_object

def chunk_document(llm_client: LLMClient, data: Dict[str, Any], min_chunk_tokens: int, embedding_model: str, filename: str) -> List[Dict[str, Any]]:
    print("[INFO] Assuming direct output structure (no 'analyzeResult' key).")
    document_tree = build_tree_from_direct_output(data)

    logging.info(f"Document tree: {document_tree}")
    text_chunks = process_tree_and_chunk(document_tree[0].children, min_words=min_chunk_tokens)

    data_objects = []
    chunk_number = 1
    for chunk_text in text_chunks:
        embedding = get_embedding(llm_client, chunk_text, embedding_model)
        token_count = len(chunk_text.split())
        char_count = len(chunk_text)

        data_objects.append({
            "content": chunk_text,
            "token_length": token_count,
            "char_length": char_count,
            "page_numbers": [],
            "chunk_number": chunk_number,
            "filename": filename,
            "_additional": {"vector": embedding}
        })
        chunk_number += 1

    print(f"Total number of chunks (tree-based): {len(data_objects)}")

    with open("chunks_tree_based.txt", "w", encoding="utf-8") as f_chunks:
        for i, chunk in enumerate(data_objects):
            f_chunks.write(f"--- Chunk Number: {i + 1} ---\n")
            f_chunks.write(chunk["content"])
            f_chunks.write("\n\n")
        print("✅ Tree-based chunks written to chunks_tree_based.txt")

    return data_objects