import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actions.convert_document_action import (
    ConvertDocumentAction,
    ConvertDocumentInputSchema
)

file_path = "sample.pdf"   
mode = "docling"           
with open(file_path, "rb") as f:
    document_content = f.read()
document_stream = io.BytesIO(document_content)    

input_schema = ConvertDocumentInputSchema(
    mode=mode,
    document=document_stream
)

action = ConvertDocumentAction()
output = action.execute(input_schema)

print("=== Converted Output ===")
print(output.content)
