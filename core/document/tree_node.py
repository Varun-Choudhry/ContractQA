# tree_node.py

from typing import List, Optional


class TreeNode:
    def __init__(self, node_id: str, node_type: str, text: str = "", roles: Optional[List[str]] = None, children: Optional[List["TreeNode"]] = None, table_data: Optional[List[dict]] = None):
        self.id = node_id
        self.type = node_type  # section, paragraph, table
        self.text = text
        self.roles = roles or []
        self.children = children or []
        self.table_data = table_data or []  

    def to_string(self) -> str:
        if self.type == "paragraph":
            if "title" in self.roles:
                return f"<title>{self.text}</title>"
            return self.text
        elif self.type == "table":
            return self.format_html_table()
        else:
            return "\n".join(child.to_string() for child in self.children)

    def format_html_table(self) -> str:
        html = "<table border='1'>\n"
        rows = self.get_table_row_count()
        cols = self.get_table_column_count()

        cell_map = self.get_table_cell_content_map()

        for r in range(rows):
            html += "  <tr>\n"
            for c in range(cols):
                content = cell_map.get((r, c), "")
                html += f"    <td>{content}</td>\n"
            html += "  </tr>\n"
        html += "</table>\n"
        return html

    def get_table_row_count(self) -> int:
        return self.table_data[0]["rowCount"] if self.table_data else 0

    def get_table_column_count(self) -> int:
        return self.table_data[0]["columnCount"] if self.table_data else 0

    def get_table_cell_content_map(self) -> dict:
        cell_map = {}

        for cell in self.table_data:
            row_index = cell["rowIndex"]
            col_index = cell["columnIndex"]
            content = cell["content"]
            row_span = cell.get("rowSpan", 1)
            col_span = cell.get("columnSpan", 1)

            for r in range(row_span):
                for c in range(col_span):
                    cell_map[(row_index + r, col_index + c)] = content
        return cell_map
