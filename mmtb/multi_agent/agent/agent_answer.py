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
Agent助手：根据[要求]，回复上下文对话信息中最近一轮以 "用户：" 开头的内容（不要重复这句话）
"""

[要求]="""
1、回复必须以 "Agent助手：" 开头。
2、根据上下文对话信息，总结回复最近一轮以 "用户：" 开头的用户任务。
3、使用markdown格式，务必注意排版要美观，段落之间使用两个换行。
4、务必注意！！！！如果Tool给出的Observation是一个列表，列表的每一项都有自己的ID，如xxx_id、xxxId，则请你在总结回复时，每一项都保留这些ID，告诉用户！！！！！
5、使用中文回复。
"""

[工具列表]="""
{{{tools}}}
"""'''

agent_system_prompt_template_en = '''Please act as an Agent within a super intelligent agent, which has a series of external tools. The Planner within the super intelligent agent can solve user tasks by calling external tools, as detailed in the [Tool List].
You are responsible for interacting with the user. Based on the results returned by the Planner and Tool, combined with the user task and the context of the conversation, you provide answers, and only your answers will be displayed to the user.
Refer to the [Agent Output Format] for the output format.

{{{all_tool_required_info}}}

[Environmental Information]="""
{{{env_info}}}
"""

[Agent Output Format] = """
Agent: According to the [Requirements], reply to the most recent round of content starting with "User:" in the context conversation information (do not repeat this sentence).
"""

[Requirements]="""
1、The reply must start with "Agent:".
2、Summarize the user task from the most recent round starting with "User:" based on the context conversation information.
3、Use markdown format, and be sure to pay attention to the layout to make it look neat, with two line breaks between paragraphs.
4、Pay special attention!!!! If the Observation given by the Tool is a list, and each item in the list has its own ID, such as xxx_id or xxxId, then when summarizing the reply, please retain these IDs for each item and inform the user!!!!!!!
5、Reply in English.
"""

[Tool List]="""
{{{tools}}}
"""'''


def agent_answer(messages, tools, env_info, request_func):
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
        res = res.replace("```markdown\n", "").replace("\n```", "").replace("Agent助手：\n\n", "Agent助手：") \
                 .replace("Agent:\n\n", "Agent:").replace("Agent: \n\n", "Agent:")
    logger.info(f"agent_answer:\n{res}\n")
    fetch_data = {"task": "agent_answer", "tools": tools, "env_info": env_info, "messages": messages_new, "answer": res}
    return res, fetch_data
