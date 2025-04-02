import json
import os

from utils import logger


tool_system_prompt_template_zh = '''请你扮演一个超级智能体中的外部工具Tool，你的这些外部工具可以用来解决用户任务，具体见[工具列表]。
请你根据超级智能体的Planner输出的工具名和输入参数，模拟工具的执行结果。如果Planner给出的Action_List里有多个工具，则请你分别进行模拟，数量要与Action_List的数量一致，结果存在Observation_List中。
输出格式参考[Tool输出格式]。

[环境信息]="""
{{{env_info}}}
"""

[Tool输出格式]="""
Tool：
```json
{
    "Observation_List": [
        {
            "status_code": "参考[工具调用结果要求]，给出包含 HTTP 响应状态代码",
            "response": "参考[工具调用结果要求]，模拟执行动作的结果。确保您的响应采用 JSON 格式、包含真实数据并符合 OpenAPI 规范格式。"
        }
    ]
}
```
"""

[工具调用结果要求]="""
1. 根据 OpenAPI 规范验证请求中的 HTTP 方法和参数。
2. 生成严格遵循 OpenAPI 规范中指定格式的响应，并确保其为 JSON 格式。
3. 响应应包含真实数据，避免使用占位符。
4. 通过提供适当的错误响应来处理边缘情况。
5. 对于没有长度限制的请求，如get方法，请确保在响应中返回 3～5 个样本，务必注意不能使用省略符号！！！！！如// xxx、...等来省略样本信息，需要符合 JSON 格式，否则会导致 JSON 无法解析！！！！！
6. 尽量使用中文模拟响应。
"""

[工具列表]="""
{{{tools}}}
"""
'''

tool_system_prompt_template_en = '''Please act as an external tool, Tool, within a super intelligent agent. These external tools can be used to solve user tasks, as detailed in the [Tool List].
Based on the tool name and input parameters output by the super intelligent agent's Planner, simulate the execution results of the tool.
If there are multiple tools in the Action_List provided by the Planner, please simulate each one separately, ensuring the number matches the Action_List, and store the results in the Observation_List.
Refer to the [Tool Output Format] for the outputformat.

[Environment Information]="""
{{{env_info}}}
"""

[Tool Output Format]="""
Tool:
```json
{
    "Observation_List": [
        {
            "status_code": "Refer to [Tool Invocation Result Requirements] for the HTTP response status code",
            "response": "Refer to [Tool Invocation Result Requirements] to simulate the result of the action execution. Ensure your response is in JSON format, contains real data, and complies with the OpenAPI specification format."
        }
    ]
}
```
"""

[Tool Invocation Result Requirements]="""
1. Validate the HTTP method and parameters in the request according to the OpenAPI specification.
2. Generate a response that strictly follows the format specified in the OpenAPI specification and ensure it isin JSON format.
3. The response should contain real data, avoiding the use of placeholders.
4. Handle edge cases by providing appropriate error responses.
5. For requests without length limitations, such as the GET method, ensure the response returns 3 to 5 samples, and be careful not to use ellipses like// xxx, ... to omit sample information, as it must conform to JSON format, otherwise it will cause JSON parsing errors!!!!!!!
6. Try to simulate responses in English.
"""

[Tool List]="""
{{{tools}}}
"""'''


def tool(messages, tools, env_info, request_func):
    language = os.getenv("LANGUAGE")
    if language == "zh":
        tool_system_prompt_template = tool_system_prompt_template_zh
    else:
        tool_system_prompt_template = tool_system_prompt_template_en
    tool_system_prompt = tool_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                    .replace("{{{env_info}}}", env_info)
    # print(tool_system_prompt)
    messages_new = [
        {
            "role": "system",
            "content": tool_system_prompt
        }
    ]
    messages_new.extend(messages)
    res = request_func(messages_new)
    logger.info(f"tool:\n{res}\n")
    fetch_data = {"task": "tool", "tools": tools, "env_info": env_info, "messages": messages_new, "answer": res}
    return res, fetch_data
