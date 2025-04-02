import json
import os

from utils import remove_prepare_ask_tools, random_select_answer_cot, get_all_tool_info, logger


user_system_prompt_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体具备一系列外部工具，可以使用外部工具解决你提出的任务。
接下来，请你根据[要求]提出3个你需要超级智能体解决的任务。
这3个任务都需要组合使用[工具列表]里的工具(包括：{{{all_tool_name}}})才能够完成，需要具体、多样、需要串行调用多个工具来解决任务。
最后请你按照[格式]输出最终结果，不要生成多余的文字。

{{{all_tool_required_info}}}

[要求]="""
1、用户任务的描述里必须包含调用工具所需的所有必填参数的信息，其他的非必填参数的信息，请你看情况添加，使用自然语言描述。
2、用户任务需要使用不同种类的句子结构：祈使句、陈述句、疑问句等。
3、用户任务应该包含不同的语气：口语化、正式、礼貌、直接等。
4、确保用户任务的长度各不相同，有短到长，长度逐渐递增。
5、确保用户任务涉及不同的主题/实例，不同的场景，不同的角色身份。
6、根据[工具列表]中所有工具的description，请你提取在所有description中出现的共同实体，并确保用户任务中出现该实体。
7、务必不要在用户任务中明确指定要使用的工具名。
8、调用的多个工具之间必须有依赖关系，调用之间的依赖关系是指，必须在调用工具A完成之后才能运行调用工具B，即调用工具B之前必须先调用工具A。
9、任务难度分为easy、medium、hard三个等级，easy代表简单，medium代表中等，hard代表困难，更难的任务需要更多的步骤执行，确保你生成的3个任务中，都是中等难度以上的任务。
"""

[工具列表]="""
{{{tools}}}
"""

[格式]="""
```json
{
    "任务1": {
        "任务描述": "xxx",
        "任务难度": "medium|hard",
        "解决任务的整体规划": "请你给出解决用户任务的整体规划，包括每一步骤需要调用哪个工具，请你首先串行给出要调用的工具"
    },
    "任务2": {
        "任务描述": "xxx",
        "任务难度": "medium|hard",
        "解决任务的整体规划": "请你给出解决用户任务的整体规划，包括每一步骤需要调用哪个工具，请你首先串行给出要调用的工具"
    },
    "任务3": {
        "任务描述": "xxx",
        "任务难度": "medium|hard",
        "解决任务的整体规划": "请你给出解决用户任务的整体规划，包括每一步骤需要调用哪个工具，请你首先串行给出要调用的工具"
    }
}
```
"""'''

user_system_prompt_template_en = '''Please act as a user who is interacting with a super intelligent agent.
This super intelligent agent has access to a range of external tools and can use these tools to solve the tasks you propose.
Next, based on the [Requirements], please propose 3 tasks that you need the super intelligent agent to solve.
These 3 tasks must require the combined use of tools from the [Tool List] (including: {{{all_tool_name}}}) to be completed.
The tasks should be specific, diverse, and require the sequential invocation of multiple tools to solve.
Finally, please output the final result according to the [Format] without generating any extra text.

{{{all_tool_required_info}}}

[Requirements]="""
1. The description of the user's task must include all the required parameters needed to invoke the tools, while other optional parameters can be added as you see fit, using natural language.
2. The user's tasks should use different types of sentence structures: imperative, declarative, interrogative, etc.
3. The user's tasks should include different tones: colloquial, formal, polite, direct, etc.
4. Ensure that the length of the user's tasks varies, from short to long, gradually increasing in length.
5. Ensure that the user's tasks involve different themes/instances, different scenarios, and different roles.
6. Based on the descriptions of all tools in the [Tool List], extract the common entities that appear in all descriptions and ensure that these entities appear in the user's tasks.
7. Do not explicitly specify the names of the tools to be used in the user's tasks.
8. There must be dependencies between the multiple tools invoked, meaning that tool A must be called and completed before tool B can be run, i.e., tool B must be invoked after tool A.
9. The difficulty of the tasks is divided into easy, medium, and hard levels. Easy represents simple, medium represents moderate, and hard represents difficult. Ensure that the 3 tasks you generate are all of medium difficulty or above.
"""

[Tool List]="""
{{{tools}}}
"""

[Format]="""
```json
{
    "Task 1": {
        "Task Description": "xxx",
        "Task Difficulty": "medium|hard",
        "Overall Plan to Solve the Task": "Please provide an overall plan to solve the user's task, including which tool to invoke at each step, and first sequentially list the tools to be invoked."
    },
    "Task 2": {
        "Task Description":"xxx",
        "Task Difficulty": "medium|hard",
        "Overall Plan to Solve the Task": "Please provide an overall plan to solve the user's task, including which tool to invoke at each step, and first sequentially list the tools to be invoked."
    },
    "Task 3": {
        "Task Description": "xxx",
        "Task Difficulty": "medium|hard",
        "Overall Plan to Solve the Task": "Please provide an overall plan to solve the user's task, including which tool to invoke at each step, and first sequentially list the tools to be invoked."
    }
}
```
"""'''


def user_multi_tool(messages, tools,  request_func):
    tools_ = remove_prepare_ask_tools(tools)
    all_tool_name, all_tool_required_info = get_all_tool_info(tools_)
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_system_prompt_template = user_system_prompt_template_zh
    else:
        user_system_prompt_template = user_system_prompt_template_en
    user_system_prompt = user_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                    .replace("{{{all_tool_name}}}", all_tool_name) \
                                                    .replace("{{{all_tool_required_info}}}", all_tool_required_info)
    messages_new = [
        {
            "role": "user",
            "content": user_system_prompt
        }
    ]
    res = request_func(messages_new)
    logger.info(f"user_multi_tool:\n{res}\n")
    user_task = random_select_answer_cot(res)
    logger.info(f"user_multi_tool:\n{user_task}\n")
    user_message = [{"role": "user", "content": user_task}]
    fetch_data = {"task": "user_multi_tool", "tools": tools, "env_info": None, "messages": messages_new, "answer": res}
    return user_message, fetch_data
