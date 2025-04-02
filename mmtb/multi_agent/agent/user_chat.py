import json
import os
import random

from utils import logger, remove_prepare_ask_tools


user_system_prompt_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体具备一系列外部工具，可以使用外部工具解决你提出的任务。
接下来，请你根据[要求]提出5个你需要超级智能体解决的闲聊任务。
这5个闲聊任务都不需要使用[工具列表]里的任何工具，但是主题上需要有一些相关性。
最后请你按照[格式]输出最终结果，不要生成多余的文字。

[要求]="""
1、用户任务是一个闲聊任务，必须与[工具列表]的功能无关，但是主题有一定的相关性。
2、用户任务需要使用不同种类的句子结构：祈使句、陈述句、疑问句等。
3、用户任务应该包含不同的语气：口语化、正式、礼貌、直接等。
4、确保用户任务的长度各不相同，有短到长，长度逐渐递增。
5、确保用户任务涉及不同的主题/实例，不同的场景，不同的角色身份。
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

user_system_prompt_template_en = '''Please pretend to be a user interacting with a super intelligent agent.
This super intelligent agent has a series of external tools that can be used to solve tasks you propose.
Next, based on the [Requirements], propose 5 casual conversation tasks that you need the super-intelligent agent to solve.
These 5 casual conversation tasks should not use any tools from the [Tool List], but should have some thematic relevance.
Finally, please output the final result according to the [Format], without generating any superfluous text.

[Requirements]=""
1. The user task is a casual conversation task, which must be unrelated to the functions of the [Tool List], but should have some thematic relevance.
2. User tasks need to use different types of sentence structures: imperative, declarative, interrogative, etc.
3. User tasks should include different tones: colloquial, formal, polite, direct, etc.
4. Ensure that the lengths of the user tasks are different, ranging from short to long, with gradually increasing length.
5. Ensure that the user tasks involve different themes/examples, different scenarios, and different role identities.
"""

[Tool list]=""
{{{tools}}}
"""

[Format]=""
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


def user_chat(messages, tools,  request_func):
    tools = remove_prepare_ask_tools(tools)
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_system_prompt_template = user_system_prompt_template_zh
    else:
        user_system_prompt_template = user_system_prompt_template_en
    user_system_prompt = user_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4))
    messages_new = [
        {
            "role": "user",
            "content": user_system_prompt
        }
    ]
    res = request_func(messages_new)
    logger.info(f"user_chat: {res}\n")
    user_task = parse_answer(res)
    logger.info(f"user_chat:\n{user_task}\n")
    user_message = [{"role": "user", "content": user_task}]
    fetch_data = {"task": "user_ask", "tools": tools, "env_info": None, "messages": messages_new, "answer": res}
    return user_message, fetch_data
