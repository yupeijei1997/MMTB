import json
import uuid
import ast

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages, AstVisitor


class WattMultiTurnMessages(SimulateMultiTurnMessages):
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []
        self.timeout = 300
    
    def preprocess_to_simple(self, messages):
        if len(self.model_messages) == 0:
            self.model_messages = remove_messages(messages, is_english=self.is_english)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages += remove_messages(
                    [{"role": "user", "content": messages[-1]["content"]}],
                    is_english=self.is_english
                )
            elif messages[-1]["role"] == "tool":
                assistant = None
                observation = []
                idx = -1
                while True or idx > -len(messages):
                    if messages[idx]["role"] == "assistant":
                        assistant = messages[idx]
                        break
                    if messages[idx]["role"] == "tool":
                        observation.append(messages[idx])
                    idx -= 1
                idmap_observation = {}
                assert len(observation) == len(assistant["tool_calls"])
                for tool_call in assistant["tool_calls"]:
                    idmap_observation[tool_call["id"]] = tool_call["function"]["name"]
                for obser in observation:
                    assert obser["tool_call_id"] in idmap_observation
                    self.model_messages.append({
                        "role": "tool", "name": idmap_observation[obser["tool_call_id"]],
                        "content": obser["content"]
                    })
        return self.model_messages
    
    def parameters2arguments(self, function_dict):
        return {
            "name": function_dict["name"],
            "arguments": function_dict["parameters"] if "parameters" in function_dict else function_dict["arguments"] 
        }

    def post_process_tool_call(self, answer):
        text = None
        tool_calls = None
        try:
            if answer.startswith("[") and answer.endswith("]"):
                try:
                    self.model_messages.append({"role":"assistant", "content": answer})
                    astor = AstVisitor()
                    astor.visit(ast.parse(answer))
                    answer_ = astor.function
                    text = "use {} to solve user problem".format(", ".join([_["name"] for _ in answer_]))
                    tool_calls = [{"id":str(uuid.uuid4()), "function":_} for _ in answer_]
                except Exception as e:
                    print(f"process error: {e}")
                    pass
            else:
                self.model_messages.append({"role":"assistant", "content": answer})
                text = answer
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None


def main():
    handle = WattMultiTurnMessages("http://111.111.111.111:12345")
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
