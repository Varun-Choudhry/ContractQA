from typing import List
import boto3
import json

class BedrockClient:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def embed_text(self, text: str, model: str) -> List[float]:
        print(f"Calling Bedrock embedding API with model: {model}")
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body["embedding"]

    def generate_text(self, prompt: str, model: str, system_prompt: str = None, **kwargs) -> str:
        
        print(f"Calling Bedrock text generation API with model: {model}")
        body = json.dumps({"prompt": f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:" if system_prompt else f"Human: {prompt}\n\nAssistant:", **kwargs})
        response = self.client.invoke_model(
            body=body,
            modelId=model,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body.get("completion", response_body.get("generations", [{}])[0].get("text", "")).strip()
