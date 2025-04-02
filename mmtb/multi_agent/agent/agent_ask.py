import json
import os

from utils import get_all_tool_info, logger


agent_system_prompt_template_zh = '''请你扮演一个超级智能体中的Agent助手，超级智能体拥有一系列外部工具，超级智能体中的Planner可以通过调用外部工具来解决用户任务，具体见[工具列表]。
你负责与用户进行交互，你根据Planner和Tool返回的结果，结合用户任务以及上下文对话信息进行回答，只有你的回答会展示给用户。
输出格式参考[Agent助手输出格式]。

{{{all_tool_required_info}}}

[环境信息]="""
{{{env_info}}}
"""

[Agent助手输出格式]="""
Agent助手：根据[要求]，向用户提出认为需要用户输入或补充的信息（不要重复这句话）
"""

[要求]="""
1、回复必须以 "Agent助手：" 开头。
2、根据上下文对话信息，尤其是Planner中的信息，结合工具中的required参数，向用户提出认为需要用户输入或补充的信息，注意提问时不要包括参数的名字。
3、使用markdown格式，务必注意排版要美观，段落之间使用两个换行。
4、使用中文回复。
"""

[工具列表]="""
{{{tools}}}
"""'''


agent_system_prompt_template_en = '''You are to act as an Agent assistant with in a super intelligent agent. The super intelligent agent has a series of external tools, and the Planner within the system can solve user tasks by invoking these external tools, as detailed in the [Tool List].
You are responsible for interacting with users, and you provide answers based on the results returned by the Planner and Tools, combined with the user's task sand contextual dialogue information. Only your responses will be displayed to the user.
The output format should refer to the [Agent Assistant OutputFormat].

{{{all_tool_required_info}}}

[Environment Information]="""
{{{env_info}}}
"""

[Agent Assistant Output Format]="""
Agent Assistant: Based on the [Requirements], ask the user for any information or input you think is necessary (do not repeat this sentence).
"""

[Requirements]="""
1. The response must start with "Agent Assistant:".
2. Based on the contextual dialogue information, especially the information from the Planner, and combined with the required parameters from the tools, ask the user for any information or input you think is necessary, ensuring not to include the parameter names in your questions.
3. Use markdown format, ensuring the layout is aesthetically pleasing, with two line breaks between paragraphs.
4. Respond in English.
"""

[Tool List]="""
{{{tools}}}
"""'''

def agent_ask(messages, tools, env_info, request_func):
    all_tool_name, all_tool_required_info = get_all_tool_info(tools)
    language = os.getenv("LANGUAGE")
    if language == "zh":
        agent_system_prompt_template = agent_system_prompt_template_zh
    else:
        agent_system_prompt_template = agent_system_prompt_template_en
    agent_system_prompt = agent_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                      .replace("{{{env_info}}}", env_info) \
                                                      .replace("{{{all_tool_name}}}", all_tool_name) \
                                                      .replace("{{{all_tool_required_info}}}", all_tool_required_info)
    messages_new = [
        {
            "role": "system",
            "content": agent_system_prompt
        }
    ]
    messages_new.extend(messages)
    res = request_func(messages_new)
    if "```markdown\n" in res:
        res = res.replace("```markdown\n", "").replace("\n```", "")
    logger.info(f"agent_ask:\n{res}\n")
    fetch_data = {"task": "agent_ask", "tools": tools, "env_info": env_info, "messages": messages_new, "answer": res}
    return res, fetch_data
