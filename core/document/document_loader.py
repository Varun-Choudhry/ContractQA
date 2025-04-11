import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

def load_document(endpoint: str, key: str, file_path: str) -> dict:
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", body=f)
        result = poller.result()
        return result.as_dict()

def load_document_from_upload(endpoint: str, key: str, uploaded_file) -> dict:
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", body=uploaded_file)
    result = poller.result()
    print(poller.result())
    return result.as_dict()