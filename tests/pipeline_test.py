import yaml
import io
import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator import Orchestrator


def load_sample_document():
    with open("sample.pdf", "rb") as f:
        document_content = f.read()
    document_stream = io.BytesIO(document_content)    
    return document_stream 

pipeline = [
    "convert_document_action",
    "chunker_action",
    "embedder_action",
    "vector_upload_action"
]

def test_pipeline():
    first_input = {
        "document": load_sample_document(),
        "mode": "docling", 
    }
    
    orchestrator = Orchestrator(pipeline=pipeline, first_input=first_input)

    final_output = orchestrator.run()

    print("Final Output:", final_output)

if __name__ == "__main__":
    test_pipeline()
