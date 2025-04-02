import json
import random
import os

from utils import logger, remove_prepare_ask_tools


user_system_prompt_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体具备一系列外部工具，可以使用外部工具解决你提出的任务。
接下来，请你根据[要求]提出5个你需要超级智能体解决的模糊不清的任务。
这5个任务都需要使用[工具列表]里的{{{tool}}}才能够完成，但是会让超级智能体不清楚如何填写{{{tool}}}里的某些必填(required)参数，需要多样。
最后请你按照[格式]输出最终结果，不要生成多余的文字。

工具{{{tool}}}的必填参数有：{{{tool_required}}}，非必填参数有：{{{tool_no_required}}}

[要求]="""
1、用户任务的描述里必须缺乏调用{{{tool}}}时所需的所有必填参数的信息，剩下的非必填参数的信息，请你看情况添加，使用自然语言描述。
注意工具参数允许一定的参数推导，即根据用户任务描述可以推导出工具参数的话，就不算缺乏了必要信息，缺乏指的是即使通过推导也无法获得参数值。
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

user_system_prompt_template_en = '''Please act as a user who is interacting with a super intelligent agent.
This super intelligent agent has access to a range of external tools and can use these tools to solve the tasks you propose.
Next, based on the [Requirements], please propose 5 ambiguous tasks that you need the super intelligent agent to solve.
All 5 tasks must require the use of {{{tool}}} from the [Tool List] to be completed, but will leave the super intelligent agent unclear on how to fill in some of the required parameters of {{{tool}}}, and should be diverse.
Finally, please output the final result according to the [Format], without generating any extra text.

The required parameters for tool {{{tool}}} are: {{{tool_required}}}, and the optional parameters are: {{{tool_no_required}}}

[Requirements]="""
1. The description of the user's task must lack all the necessary information for calling {{{tool}}}, leaving only the optional parameter information, which you can add as you see fit, using natural language descriptions.
Note that tool parameters allow for some parameter inference, meaning that if the tool parameters can be inferred from the user's task description, it does not count as lacking necessary information. Lacking means that even through inference, the parameter values cannot be obtained.
2. The user's tasks need touse different types of sentence structures: imperative sentences, declarative sentences, interrogative sentences, etc.
3. The user's tasks should include different tones: colloquial, formal, polite, direct, etc.
4. Ensure that the length of the user's tasks varies, from short to long, gradually increasing in length.
5. Ensure that the user's tasks involve different themes/instances, different scenarios, and different roles.
6. Based on the descriptions of all tools in the [Tool List], extract the common entities that appear in all descriptions and ensure that these entities appear in the user's tasks.
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

def parse_answer(user_tasks):
    user_tasks = user_tasks.replace("```json\n", "").replace("\n```", "")
    user_tasks = json.loads(user_tasks)
    task_keys = list(user_tasks.keys())
    task_key = random.choice(task_keys)
    user_task = user_tasks[task_key]
    user_task = "用户：" + user_task
    return user_task


def user_ask(messages, tools,  request_func):
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
    logger.info(f"user_ask: {res}\n")
    user_task = parse_answer(res)
    logger.info(f"user_multi_tool:\n{user_task}\n")
    user_message = [{"role": "user", "content": user_task}]
    fetch_data = {"task": "user_ask", "tools": tools, "env_info": None, "messages": messages_new, "answer": res}
    return user_message, fetch_data
