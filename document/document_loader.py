from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
    
def convert_document(document="    ", mode="docling", config):
    if mode == "azure":
        response = convert_with_azure(document,config)
        print(type(response))
        return response
    elif mode == "docling":
        response = convert_with_docling(document,config)
        print(type(response))
        
        return response
        
        
def convert_with_docling(document,config):
    source = DocumentStream(name="my_doc.pdf", stream=document)
    converter = DocumentConverter()
    result = converter.convert(source)
    
    return result.document.export_to_markdown()
    
def covert_with_azure(document,config):
    #need to define init
    endpoint=""
    key=""
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", body=document,output_content_format=DocumentContentFormat.MARKDOWN,)
    result = poller.result()
    return result.content_format()    
    