import json
import uuid

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages


class XLAMMultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []

    def preprocess_to_simple(self, messages):
        if len(self.model_messages) == 0:
            self.model_messages = remove_messages(messages, is_english=True)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages.append({"role": "user", "content": messages[-1]["content"].replace("用户：", "").replace("User:", "").strip()})
            elif messages[-1]["role"] == "tool":
                self.model_messages.append({"role": "tool", "content": messages[-1]["content"]})
        return self.model_messages

    def post_process_tool_call(self, answer):
        try:
            if "tool_calls" in answer:
                try:
                    answer = json.loads(answer)
                except Exception as e:
                    print(f"json loads error: {e}")
                    pass

            if "tool_calls" in answer and type(answer) == dict:
                text = "use {} to solve user problem".format(
                    ", ".join([
                        _["name"] for _ in answer["tool_calls"]
                    ])
                )
                tool_calls = [{"id": str(uuid.uuid4()), "function": _} for _ in answer["tool_calls"]]
                self.model_messages.append({"role": "assistant", "content": text, "tool_calls": answer["tool_calls"]})
            else:
                self.model_messages.append({"role": "assistant", "content": answer})
                text = answer
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None


def main():
    handle = XLAMMultiTurnMessages("http://111.111.111.111:12345")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in the two cities of Boston and San Francisco?"
        }
    ]
    content, tool_calls = handle.request_funcall(messages, tools)
    print(content)
    print(json.dumps(tool_calls, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()