import json
import re
import uuid

from .basic_handle import SimulateMultiTurnMessages
from .tools import remove_messages


class QwenMultiTurnMessages(SimulateMultiTurnMessages):
    # https://qwen.readthedocs.io/en/latest/framework/function_call.html#hugging-face-transformers
    def __init__(self, model_url, is_english=False):
        super().__init__(model_url, is_english)
        self.model_messages = []
        self.timeout = 300

    def add_date_to_message(self, message, env_info=None):
        if env_info is not None:
            system_content = message[0]['content'] if message[0]["role"] == "system" else "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            if self.is_english:
                system_content = system_content[:system_content.rfind("\n\nCurrent Date:")] + "\n\nCurrent Date:" + self.add_weekday_date(env_info)
            else:
                system_content = system_content[:system_content.rfind("当前日期：")] + "\n\n当前日期：" + self.add_weekday_date(env_info)
            if message[0]["role"] == "system":
                message[0]["content"] = system_content.strip()
            else:
                message.insert(0, {"role": "system", "content": system_content.strip()})
            return message
        else:
            return message

    def try_parse_tool_calls(self, content: str):
        """Try parse the tool calls."""
        tool_calls = []
        offset = 0
        for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
            if i == 0:
                offset = m.start()
            try:
                func = json.loads(m.group(1))
                tool_calls.append({"type": "function", "function": func})
                if isinstance(func["arguments"], str):
                    func["arguments"] = json.loads(func["arguments"])
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
                pass

        if tool_calls:
            if offset > 0 and content[:offset].strip():
                c = content[:offset]
            else: 
                c = ""

            return {"role": "assistant", "content": c, "tool_calls": tool_calls}

        return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

    def preprocess_to_simple(self, messages):
        if len(self.model_messages) == 0:
            self.model_messages = remove_messages(messages, is_english=self.is_english)
        else:
            if messages[-1]["role"] == "user":
                self.model_messages += remove_messages(
                    [{"role":"user", "content": messages[-1]["content"]}],
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
    
    def post_process_tool_call(self, answer):
        try:
            assistant_message = self.try_parse_tool_calls(answer)
            self.model_messages.append(assistant_message)
            text = assistant_message.get("content", None)
            if "tool_calls" in assistant_message:
                tool_calls = [{
                    "id": str(uuid.uuid4()), "function": tool_call["function"]
                } for tool_call in assistant_message.get("tool_calls", None)]
            else:
                tool_calls = None

            return text, tool_calls

        except Exception as e:
            print(f"error: {e}")
            return None, None


def main():
    handle = QwenMultiTurnMessages("http://111.111.111.111:12345")
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
