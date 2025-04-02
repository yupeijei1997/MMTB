import json
import random
import os

from utils import remove_prepare_ask_tools, random_select_answer, logger


user_system_prompt_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体具备一系列外部工具，可以使用外部工具解决你提出的任务。
接下来，请你根据[要求]提出5个你需要超级智能体解决的任务。
这5个任务都需要使用[工具列表]里的{{{tool}}}才能够完成，且都只需要调用{{{tool}}}一次，需要具体、多样。
最后请你按照[格式]输出最终结果，不要生成多余的文字。

工具{{{tool}}}的必填参数有：{{{tool_required}}}，非必填参数有：{{{tool_no_required}}}

[要求]="""
1、用户任务的描述里必须包含调用{{{tool}}}所需的所有必填参数的信息，其他的非必填参数的信息，请你看情况添加，使用自然语言描述。
2、用户任务需要使用不同种类的句子结构：祈使句、陈述句、疑问句等。
3、用户任务应该包含不同的语气：口语化、正式、礼貌、直接等。
4、确保用户任务的长度各不相同，有短到长，长度逐渐递增。
5、确保用户任务涉及不同的主题/实例，不同的场景，不同的角色身份。
6、根据[工具列表]中所有工具的description，请你提取在所有description中出现的共同实体，并确保用户任务中出现该实体。
7、务必不要在用户任务中明确指定要使用的工具{{{tool}}}。
"""

[工具列表]="""
{{{tools}}}
"""

[格式]="""
```json
{
    "任务1": "xxx",
    "任务2": "xxx",
    "任务3": "xxx",
    "任务4": "xxx",
    "任务5": "xxx"
}
```
"""'''

user_system_prompt_template_en = '''Please act as a user interacting with a super intelligent agent.
This super intelligent agent has access to a range of external tools and can use these tools to solve the tasks you propose.
Next, please propose 5 tasks that you need the super intelligent agent to solve based on the [Requirements].
All 5 tasks must require the use of {{{tool}}} from the [Tool List] to be completed, and each task should only require a single call to {{{tool}}}.
The tasks should be specific and diverse.
Finally, please output the final result according to the [Format] without generating any extra text.

The required parameters for tool {{{tool}}} are: {{{tool_required}}}, and the optional parameters are: {{{tool_no_required}}}.

[Requirements]="""
1. The description of the user's task must include information on all the required parameters needed to call {{{tool}}}. For other optional parameters, please add them as you see fit, using natural language.
2. The user's tasks should use different types of sentence structures: imperative, declarative, interrogative, etc.
3. The user's tasks should include different tones: colloquial, formal, polite, direct, etc.
4. Ensure that the length of the user's tasks varies, gradually increasing from short to long.
5. Ensure that the user's tasks involve different themes/instances, different scenarios, and different roles.
6. Extract common entities that appear in all descriptions from the [Tool List] and ensure that these entities appear in the user's tasks.
7. Do not explicitly specify the tool {{{tool}}} in the user's tasks.
"""

[Tool List]="""
{{{tools}}}
"""

[Format]="""
```json
{
    "Task 1": "xxx",
    "Task 2": "xxx",
    "Task 3": "xxx",
    "Task 4": "xxx",
    "Task 5": "xxx"
}
```
"""'''

def user_single_tool(messages, tools,  request_func):
    tools_ = remove_prepare_ask_tools(tools)
    tool = random.choice(tools_)
    tool_name = tool["function"]["name"]
    tool_required = tool["function"]["parameters"]["required"]
    tool_required = ", ".join(tool_required)
    tool_all_properties = list(tool["function"]["parameters"]["properties"].keys())
    tool_no_required = []
    for property in tool_all_properties:
        if property not in tool_required:
            tool_no_required.append(property)
    tool_no_required = ", ".join(tool_no_required)
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_system_prompt_template = user_system_prompt_template_zh
    else:
        user_system_prompt_template = user_system_prompt_template_en
    user_system_prompt = user_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                    .replace("{{{tool}}}", tool_name) \
                                                    .replace("{{{tool_required}}}", tool_required) \
                                                    .replace("{{{tool_no_required}}}", tool_no_required)
    messages_new = [
        {
            "role": "user",
            "content": user_system_prompt
        }
    ]
    res = request_func(messages_new)
    logger.info(f"user_single_tool:\n{res}\n")
    user_task = random_select_answer(res)
    logger.info(f"user_single_tool:\n{user_task}\n")
    user_message = [{"role": "user", "content": user_task}]
    fetch_data = {"task": "user_single_tool", "tools": tools, "env_info": None, "messages": messages_new, "answer": res}
    return user_message, fetch_data
