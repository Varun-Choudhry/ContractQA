from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
    
def convert_document(document, mode):
    if mode == "azure":
        return convert_with_azure(document)
    elif mode == "docling":
        return convert_with_docling(document)
        
        
def convert_with_docling(document):
    source = DocumentStream(name="my_doc.pdf", stream=document)
    converter = DocumentConverter()
    result = converter.convert(source)
    return result.document.export_to_markdown()
    
def covert_with_azure(document):
    #need to define init
    endpoint=""
    key=""
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", body=document,output_content_format=DocumentContentFormat.MARKDOWN,)
    result = poller.result()
    return result.content_format()    
    