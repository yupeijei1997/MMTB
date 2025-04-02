import json
import os

from openai import OpenAI, AzureOpenAI


class GPTAZUREMultiTurnMessages:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    def request_model(self, messages):
        kwargs = {
            "messages": messages,
            "temperature": 0.1,
            "timeout": 300,
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT")
        }
        api_response = self.client.chat.completions.create(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
        return text


class GPTMultiTurnMessages:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def request_model(self, messages):
        kwargs = {
            "messages": messages,
            "temperature": 0.1,
            "timeout": 300,
            "model": os.getenv("OPENAI_MODEL")
        }
        api_response = self.client.chat.completions.create(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
        return text


def main():
    handle = GPTMultiTurnMessages()
    messages = [
        {
            "role": "user",
            "content": "Hello, who are you?"
        }
    ]
    print(json.dumps(messages, ensure_ascii=False, indent=4))
    print("---")
    result = handle.request_model(messages)
    print(result)


if __name__ == "__main__":
    main()
