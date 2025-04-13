class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("The 'anthropic' library is not installed. Please install it using 'pip install anthropic'.")


    def generate_text(self, prompt: str, model: str, system_prompt: str = None, **kwargs) -> str:
        print(f"Calling Anthropic text generation API with model: {model}")
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        response = self.client.messages.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response.content[0].text.strip()