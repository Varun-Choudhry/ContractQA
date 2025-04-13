import instructor
from instructor import from_openai
from openai import AzureOpenAI
from config.config import config
import datetime
import json

LOG_PATH = "llm_calls_log.txt"

def log_llm_call(messages, response):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n--- LLM CALL @ {datetime.datetime.now()} ---\n")
            f.write(">> INPUT MESSAGES:\n")
            f.write(json.dumps(messages, indent=2))
            f.write("\n\n>> RESPONSE:\n")
            f.write(json.dumps(response, indent=2))
            f.write("\n" + "-" * 50 + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log LLM call: {e}")


def get_llm_client():
    api_version = config.get("azure_openai_chat_api_version")
    endpoint = config.get("azure_openai_chat_endpoint")
    subscription_key = config.get("azure_openai_chat_key")

    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

   
    wrapped_client = from_openai(client, mode=instructor.Mode.MD_JSON)

    
    original_create = wrapped_client.chat.completions.create

    def logging_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        messages = kwargs.get("messages") or (args[0]["messages"] if args else [])
        log_llm_call(messages, response.model_dump())
        return response

   
    if not getattr(wrapped_client.chat.completions.create, "_is_logged", False):
        logging_create._is_logged = True
        wrapped_client.chat.completions.create = logging_create

    return wrapped_client
