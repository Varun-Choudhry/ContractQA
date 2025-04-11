# core/llm/atomic_llm.py

import instructor
from instructor import from_openai
from openai import OpenAI
from config.config import config
import datetime
import json
import os

LOG_PATH = "llm_calls_log.txt"

# --- Helper: Logging ---
def log_llm_call(messages, response):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n--- LLM CALL @ {datetime.datetime.now()} ---\n")
        f.write(">> INPUT MESSAGES:\n")
        f.write(json.dumps(messages, indent=2))
        f.write("\n\n>> RESPONSE:\n")
        f.write(json.dumps(response, indent=2))
        f.write("\n" + "-" * 50 + "\n")

# --- Singleton-like client for atomic agents ---
def get_llm_client():
    client = OpenAI(
        base_url=config.get("lm_studio_url"),
        api_key="lm-studio"
    )

    wrapped_client = from_openai(client, mode=instructor.Mode.MD_JSON)

    # Wrap the chat.completions.create method
    original_create = wrapped_client.chat.completions.create

    def logging_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        try:
            # For logging
            messages = kwargs.get("messages") or (args[0]["messages"] if args else [])
            log_llm_call(messages, response.model_dump())
        except Exception as e:
            print(f"⚠️ Failed to log LLM call: {e}")
        return response

    wrapped_client.chat.completions.create = logging_create
    return wrapped_client
