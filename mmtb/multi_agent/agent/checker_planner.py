import json
import re
import os

from utils import get_all_tool_info, get_all_tool_info_for_checker, parse_answer, logger


checker_serial_system_prompt_template_zh = '''当前场景下有一个用户和一个超级智能体，该超级智能体拥有一系列外部工具，具体见[工具列表]。用户提出任务，超级智能体使用外部工具帮助用户解决任务。
超级智能体中的Planner可以通过调用外部工具来解决用户任务，Tool负责执行这些外部工具，Agent助手负责与用户进行交互，Checker负责检查各个Planner、Tool、Agent助手输出的结果是否正确。

请你扮演这个超级智能体中的Checker_Planner，你负责判断Planner的输出是否正确，你需要判断Planner输出的多工具调用计划是否合理。输出格式参考[Checker_Planner输出格式]。

[环境信息]="""
{{{env_info}}}
"""

[Checker_Planner输出格式]="""
Checker_Planner：
```json
{
    "tool_call_analysis": "根据[要求]，对Planner给出的多工具调用计划进行分析，主要对工具的串行和并行性进行分析",
    "correct": "如果Planner的计划合理，并且工具的串行和并行性没有问题，则输出yes，否则输出no"
}
```
"""

[要求]="""
参考[工具列表]里工具的定义，分析Planner输出的计划是否合理，主要分析：
1、如果Planner给出的计划是并行调用多个工具：即在Action_List里给出了多个需要并行执行的工具，你需要分析这些工具是否可以同时执行，能够同时执行是指没有参数依赖和逻辑依赖。
1）参数依赖：比如调用工具C之前必须先调用工具B获取结果作为输入参数，调用工具B之前又必须调用工具A获取结果作为输入参数，那么就需要先调用完成工具A获得结果，作为工具B的输入参数进行调用，调用完成工具B获得结果后，作为工具C的输入参数进行调用，即请你串行调用工具A、B、C。
2）逻辑依赖：即使调用工具A、B、C之间没有参数关系依赖，但是存在逻辑上的依赖，比如逻辑上调用工具C之前必须先调用工具B，调用工具B之前又必须先调用工具A，则请你也串行调用工具A、B、C。

2、如果Planner给出的计划是串行调用多个工具，请你根据参数依赖和逻辑依赖，分析这些工具是否可以并行(同时)执行。

3、如果Planner给出的计划既包含并行调用也包含串行调用，请你根据上述2个原则分别进行分析判断。
最后，请你注意，如果Planner给出的计划是串行调用多个工具，但是当前的Action_List只给出了当前步骤执行的工具，这是正确的，不算错误。
因为Action_List给出的是当前步骤要执行的动作，需要串行调用的剩余工具应该在后续步骤的Action_List里给出。
"""

{{{all_tool_required_info}}}

[工具列表]="""
{{{tools}}}
"""'''

checker_serial_system_prompt_template_en = '''In the current scenario, there is a user and a super intelligent agent equipped with a series of external tools, as detailed in the [Tool List]. The user presents a task, and the super intelligent agent uses the external tools to help the user solve the task.
The Planner within the super intelligent agent can solve user tasks by invoking external tools. The Tool is responsible for executing these external tools, the Agent Assistant is responsible for interacting with the user, and the Checker is responsible for verifying whether the results output by the Planner, Tool, and Agent Assistant are correct.

Please play the role of the Checker_Planner in this super intelligent agent. You are responsible for determining whether the Planner's output is correct. You need to judge whether the multi-tool invocation plan output by the Planner is reasonable. Refer to the [Checker_Planner Output Format] for the output format.

[Environment Information]="""
{{{env_info}}}
"""

[Checker_Planner Output Format]="""
Checker_Planner:
```json
{
    "tool_call_analysis": "Based on the [Requirements], analyze the multi-tool invocation plan provided by the Planner, focusing on the analysis of the tools' serial and parallel execution capabilities",
    "correct": "If the Planner's plan is reasonable, and there are no issues with the serial and parallel execution of the tools, then output 'yes', otherwise output 'no'"
}
```
"""

[Requirements]="""
Refer to the definitions of the tools in the [Tool List], analyze whether the plan output by the Planner is reasonable, mainly analyzing:
1. If the Planner's plan involves parallel invocation of multiple tools: that is, the Action_List specifies multiple tools to be executed in parallel, you need to analyze whether these tools can be executed simultaneously, meaning there are no parameter dependencies or logical dependencies.
   a) Parameter dependency: For example, before invoking tool C, it is necessary to first invoke tool B to obtain results as input parameters, and before invoking tool B, it is necessary to first invoke tool A to obtain results as input parameters. Therefore, you need to sequentially invoke tools A, B, and C to obtain the results in order.
   b) Logical dependency: Even if there is no parameter dependency between invoking tools A, B, and C, there may be a logical dependency. For example, logically, tool C must be invoked after tool B, and tool B must be invoked after tool A. In this case, you should also sequentially invoke tools A, B, and C.

2. If the Planner's plan involves serial invocation of multiple tools, you need to analyze, based on parameter dependencies and logical dependencies, whether these tools can be executed in parallel (simultaneously).

3. If the Planner's plan includes both parallel and serial invocations, you need to analyze and judge according to the above two principles separately.
Finally, please note that if the Planner's plan involves serial invocation of multiple tools, but the current Action_List only specifies the tool to be executed in the current step, this is correct and not an error.
This is because the Action_List specifies the actions to be executed in the current step, and the remaining tools that need to be invoked serially should be specified in the Action_List of subsequent steps.
"""

{{{all_tool_required_info}}}

[Tool List]="""
{{{tools}}}
"""
'''

def rule_checker_zh(messages, tools, env_info, tool_flag, request_func, enable_llm):
    analysis = {
        "format_analysis": "",
        "thought_and_plan_analysis": "",
        "tool_call_analysis": "",
        "parameter_input_analysis": "",
        "correct": "yes"
    }

    all_tool_name, all_tool_name_properties_name, all_tool_name_required = get_all_tool_info_for_checker(tools)
    planner_message = messages[-2]
    planner_content = planner_message["content"]
    try:
        planner_content_obj = parse_answer(planner_content)
    except Exception as e:
        analysis["format_analysis"] = f"Planner生成的格式错误，JSON无法解析，请不要增加//等注释信息，具体错误为：{e}"
        if "时间戳" in planner_content:
            analysis["parameter_input_analysis"] = f"Planner生成的参数错误，时间戳你无法根据时间自己推导生成，需要调用ask_user_for_required_parameters工具，向用户询问"
        analysis["correct"] = "no"
        rule_checker_result = f"Checker_Planner：\n```json\n{json.dumps(analysis, ensure_ascii=False, indent=4)}\n```"
        logger.info(f"rule_checker\n{rule_checker_result}\n")
        return analysis["correct"], analysis, rule_checker_result, None

    thought = planner_content_obj["Thought"]
    plan = planner_content_obj["Plan"]
    action_list = planner_content_obj["Action_List"]

    thought_and_plan_analysis = ""
    if "模拟执行" in thought:
        thought_and_plan_analysis += f"Planner生成的Thought里提到了模拟执行，注意工具模拟执行这个信息是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if "模拟执行" in plan:
        thought_and_plan_analysis += f"Planner生成的Plan里提到了模拟执行，注意工具模拟执行这个信息是不暴露出来的，生成错误，需要检查！！！！\n\n"
    # if "串行调用" in plan and len(action_list) > 1:
    #     thought_and_plan_analysis += f"Planner生成的Plan里提到了要串行调用工具，但是给出的Action_List里要调用的工具个数又为{len(action_list)}个，存在逻辑矛盾，生成错误，需要检查！！！！\n\n"
    # if "并行调用" in plan and len(action_list) == 1:
    #     thought_and_plan_analysis += f"Planner生成的Plan里提到了要并行调用工具，但是给出的Action_List里要调用的工具个数只有1个，存在逻辑矛盾，生成错误，需要检查！！！！\n\n"
    if "Checker" in thought:
        thought_and_plan_analysis += f"Planner生成的Thought里提到了Checker，注意Checker是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if "Checker" in plan:
        thought_and_plan_analysis += f"Planner生成的Plan里提到了Checker，注意Checker是不暴露出来的，生成错误，需要检查！！！！\n\n"

    if "ask_user_for_required_parameters" in thought:
        thought_and_plan_analysis += f"Planner生成的Thought里提到了调用ask_user_for_required_parameters工具，该工具是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if "ask_user_for_required_parameters" in plan:
        thought_and_plan_analysis += f"Planner生成的Plan里提到了调用ask_user_for_required_parameters工具，该工具是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if "prepare_to_answer" in thought:
        thought_and_plan_analysis += f"Planner生成的Thought里提到了调用prepare_to_answer工具，该工具是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if "prepare_to_answer" in plan:
        thought_and_plan_analysis += f"Planner生成的Plan里提到了调用prepare_to_answer工具，该工具是不暴露出来的，生成错误，需要检查！！！！\n\n"
    if thought_and_plan_analysis != "":
        analysis["thought_and_plan_analysis"] = thought_and_plan_analysis
        analysis["correct"] = "no"

    tool_call_analysis = ""
    parameter_input_analysis = ""
    if len(action_list) == 0:
        tool_call_analysis += f"Planner生成的Action_List为空列表[]，生成错误，需要检查！！！！\n\n"
        analysis["correct"] = "no"

    action_name_set = set()
    for action in action_list:
        action_name = action["name"]
        action_name_set.add(action_name)

    for action in action_list:
        action_name = action["name"]
        action_arguments = action["arguments"]
        arguments_name_list = list(action_arguments.keys())

        if action_name not in all_tool_name:
            tool_call_analysis += f"Planner生成的工具{action_name}不在[工具列表]的候选工具里，生成错误，需要检查！！！！\n\n"
            analysis["correct"] = "no"
        else:
            for argument_name, argument_value in action_arguments.items():
                if argument_name not in all_tool_name_properties_name[action_name]:
                    parameter_input_analysis += f"Planner生成的工具参数{argument_name}不在工具{action_name}的参数列表里，生成错误，需要检查！！！！\n\n"
                    analysis["correct"] = "no"

                if argument_name in all_tool_name_required[action_name] and argument_value == "":
                    parameter_input_analysis += f"Planner生成的工具参数{argument_name}为工具的必填参数，但是其值为{argument_value}，是一个空字符串，生成错误，需要检查！！！！\n\n"
                    analysis["correct"] = "no"

            for required_argument in all_tool_name_required[action_name]:
                if required_argument not in arguments_name_list:
                    parameter_input_analysis += f"Planner未生成的必填工具的必填参数{required_argument}。\n生成错误，需要检查！！！！\n可能有如下2种情况：1、用户未提供必填参数的信息，如果是该情况，需要Planner调用ask_user_for_required_parameters工具，向用户询问必填信息。\n2、用户提供了必填参数的信息，但是是Planner忘记填写了，如果是该情况，则需要Planner补上必填参数的信息。\n务必注意，请不要幻觉必填参数的信息。\n\n"
                    analysis["correct"] = "no"

        if action_name == "ask_user_for_required_parameters":
            if len(action_list) > 1 and len(action_name_set) != 1:
                tool_call_analysis += f"Planner同时生成了多个工具，其中包括ask_user_for_required_parameters工具，ask_user_for_required_parameters工具无法和其他工具并发调用，需要在之后串行调用，生成错误，需要检查！！！！\n\n"
                analysis["correct"] = "no"
                continue

            ask_required_parameter_tool_name = action_arguments["tool_name"]
            if ask_required_parameter_tool_name not in all_tool_name:
                tool_call_analysis += f"Planner生成的需要向用户询问必填参数信息的工具{ask_required_parameter_tool_name}不在[工具列表]的候选工具里，生成错误，需要检查！！！！\n\n"
                analysis["correct"] = "no"
                continue

            missing_required_parameters = action_arguments["missing_required_parameters"]
            for missing_required_parameter in missing_required_parameters:
                if missing_required_parameter not in all_tool_name_required[ask_required_parameter_tool_name]:
                    parameter_input_analysis += f"Planner生成的需要向用户询问的参数{missing_required_parameter}不是工具{ask_required_parameter_tool_name}的必填参数，生成错误，需要检查！！！！\n\n"
                    analysis["correct"] = "no"

        if action_name == "prepare_to_answer":
            if len(action_list) > 1 and len(action_name_set) != 1:
                tool_call_analysis += f"Planner同时生成了多个工具，其中包括prepare_to_answer工具，prepare_to_answer工具无法和其他工具并发调用，需要在之后串行调用，生成错误，需要检查！！！！\n\n"
                analysis["correct"] = "no"
                continue

            answer_type = action_arguments["answer_type"]
            if answer_type not in ["tool", "chat"]:
                parameter_input_analysis += "Planner生成prepare_to_answer工具的参数错误，不是tool和chat其中之一，生成错误，需要检查！！！！\n\n"
                analysis["correct"] = "no"

            if tool_flag and answer_type == "chat":
                parameter_input_analysis += "前一轮消息为工具的执行结果，但是Planner生成prepare_to_answer工具的参数为chat，这是非法的，生成错误，需要检查！！！！该情况下prepare_to_answer工具的参数只能为[工具列表]里某个工具的name，或者是tool。\n\n"
                analysis["correct"] = "no"

    fetch_data = None
    if enable_llm:
        if len(action_list) > 1 or "串行" in thought or "并行" in thought or "串行" in plan or "并行" in plan:
            llm_parallel_correct, llm_parallel_analysis, llm_parallel_checker_result, fetch_data = llm_parallel_checker(messages, tools, env_info, request_func)
            if llm_parallel_correct == "no":
                analysis["correct"] = llm_parallel_correct
                tool_call_analysis += llm_parallel_analysis["tool_call_analysis"]

            pattern = r"串行调用(\w+)工具([一二三四五六七八九十]+)次"
            matches = re.findall(pattern, plan)
            for match in matches:
                analysis["correct"] = "no"
                tool_name = match[0]
                call_times = match[1]
                tool_call_analysis += f"Planner给出的Plan里说明要串行调用{tool_name}工具{call_times}次，但是Action_List里给出的工具个数又>1，这是矛盾的，需要检查！！！！！"

    analysis["tool_call_analysis"] = tool_call_analysis
    analysis["parameter_input_analysis"] = parameter_input_analysis

    rule_checker_result = f"Checker_Planner：\n```json\n{json.dumps(analysis, ensure_ascii=False, indent=4)}\n```"
    logger.info(f"rule_checker\n{rule_checker_result}\n")
    return analysis["correct"], analysis, rule_checker_result, fetch_data


def rule_checker_en(messages, tools, env_info, tool_flag, request_func, enable_llm):
    analysis = {
        "format_analysis": "",
        "thought_and_plan_analysis": "",
        "tool_call_analysis": "",
        "parameter_input_analysis": "",
        "correct": "yes"
    }

    all_tool_name, all_tool_name_properties_name, all_tool_name_required = get_all_tool_info_for_checker(tools)
    planner_message = messages[-2]
    planner_content = planner_message["content"]
    try:
        planner_content_obj = parse_answer(planner_content)
    except Exception as e:
        analysis["format_analysis"] = f"The format generated by the Planner is incorrect, and JSON cannot be parsed. Please do not add comment information such as //, the specific error is: {e}"
        if "timestamp" in planner_content:
            analysis["parameter_input_analysis"] = f"The parameters generated by the Planner are incorrect; you cannot deduce and generate the timestamp on your own based on time. You need to use the ask_user_for_required_parameters tool to inquire with the user."
        analysis["correct"] = "no"
        rule_checker_result = f"Checker_Planner：\n```json\n{json.dumps(analysis, ensure_ascii=False, indent=4)}\n```"
        logger.info(f"rule_checker\n{rule_checker_result}\n")
        return analysis["correct"], analysis, rule_checker_result, None

    thought = planner_content_obj["Thought"]
    plan = planner_content_obj["Plan"]
    action_list = planner_content_obj["Action_List"]

    thought_and_plan_analysis = ""
    if "simulated execution" in thought:
        thought_and_plan_analysis += f"The Thought generated by the Planner mentioned simulated execution. Note that the information about tool simulation execution is not exposed. If there is an error in the generation, it needs to be checked!!!!\n\n"
    if "simulated execution" in plan:
        thought_and_plan_analysis += f"The Plan generated by the Planner mentions simulated execution. Note that the information about tool simulation execution is not disclosed. If there is an error in the generation, it needs to be checked!!!!\n\n"
    if "Checker" in thought:
        thought_and_plan_analysis += f"The Thought generated by the Planner mentions the Checker. Note that the Checker is not exposed, and if there are errors in the generation, they need tobe checked!!!!\n\n"
    if "Checker" in plan:
        thought_and_plan_analysis += f"The plan generated by the Planner mentions the Checker, note that the Checker is not exposed, there is a generation error, it needs to be checked!!!!\n\n"

    if "ask_user_for_required_parameters" in thought:
        thought_and_plan_analysis += f"The Thought generated by the Planner mentioned the use of the ask_user_for_required_parameters tool, which is not supposed to be exposed. This is an error and needs to be checked!!!!\n\n"
    if "ask_user_for_required_parameters" in plan:
        thought_and_plan_analysis += f"The Plan generated by the Planner mentions the use of the ask_user_for_required_parameters tool, which is not supposed to be exposed. This is an error and needs to be checked!!!!\n\n"
    if "prepare_to_answer" in thought:
        thought_and_plan_analysis += f"The Thought generated by the Planner mentioned calling the prepare_to_answer tool, which is not supposed to be exposed. This is an error and needs to be checked!!!!\n\n"
    if "prepare_to_answer" in plan:
        thought_and_plan_analysis += f"The Thought generated by the Planner mentioned calling the prepare_to_answer tool, which is not supposed to be exposed. This is an error and needs to be checked!!!!\n\n"
    if thought_and_plan_analysis != "":
        analysis["thought_and_plan_analysis"] = thought_and_plan_analysis
        analysis["correct"] = "no"

    tool_call_analysis = ""
    parameter_input_analysis = ""
    if len(action_list) == 0:
        tool_call_analysis += f"The Action_List generated by the Planner is an empty list [], which indicates an error and needs to be checked!!!!\n\n"
        analysis["correct"] = "no"

    action_name_set = set()
    for action in action_list:
        action_name = action["name"]
        action_name_set.add(action_name)

    for action in action_list:
        action_name = action["name"]
        action_arguments = action["arguments"]
        arguments_name_list = list(action_arguments.keys())

        if action_name not in all_tool_name:
            tool_call_analysis += f"The tool {action_name} generated by the planner is not in the list of candidate tools [Tool List], resulting in an error that needs to be checked!!!!\n\n"
            analysis["correct"] = "no"
        else:
            for argument_name, argument_value in action_arguments.items():
                if argument_name not in all_tool_name_properties_name[action_name]:
                    parameter_input_analysis += f"The tool parameter {argument_name} generated by the planner is not in the parameter list of the tool {action_name}, resulting in an error that needs to be checked!!!!\n\n"
                    analysis["correct"] = "no"

                if argument_name in all_tool_name_required[action_name] and argument_value == "":
                    parameter_input_analysis += f"The tool parameter {argument_name} generated by the planner is a required parameter for the tool, but its value is {argument_value}, which is an empty string. This is an error and needs to be checked!!!!\n\n"
                    analysis["correct"] = "no"

            for required_argument in all_tool_name_required[action_name]:
                if required_argument not in arguments_name_list:
                    parameter_input_analysis += f"The Planner did not generate the required argument {required_argument} for the tool {action_name}.\nThere is a generation error that needs to be checked!!!!\nThere may be 2 possible situations: 1) The user did not provide the information for the required argument. If this is the case, the Planner needs to call the ask_user_for_required_parameters tool to ask the user for the necessary information.\n2) The user provided the information for the required argument, but the Planner forgot to fill it in. If this is the case, then the Planner needs to supplement the information for the required argument.\nPlease be sure to pay attention and do not have any illusions about the information for the required argument.\n\n"
                    analysis["correct"] = "no"

        if action_name == "ask_user_for_required_parameters":
            if len(action_list) > 1 and len(action_name_set) != 1:
                tool_call_analysis += f"The Planner has generated multiple tools at the same time, including the ask_user_for_required_parameters tool. The ask_user_for_required_parameters tool cannot be invoked concurrently with other tools and needs to be called serially afterwards. An error has occurred and needs to be checked!!!!\n\n"
                analysis["correct"] = "no"
                continue

            ask_required_parameter_tool_name = action_arguments["tool_name"]
            if ask_required_parameter_tool_name not in all_tool_name:
                tool_call_analysis += f"The tool {ask_required_parameter_tool_name} generated by the Planner, which is used to ask users for required parameter information, is not among the candidate tools in the [Tool List], indicating a generation error that needs to be checked!!!!\n\n"
                analysis["correct"] = "no"
                continue

            missing_required_parameters = action_arguments["missing_required_parameters"]
            for missing_required_parameter in missing_required_parameters:
                if missing_required_parameter not in all_tool_name_required[ask_required_parameter_tool_name]:
                    parameter_input_analysis += f"The parameter {missing_required_parameter} that the Planner generated to ask the user is not a required parameter for the tool {ask_required_parameter_tool_name}. This is a generation error and needs to be checked!!!!\n\n"
                    analysis["correct"] = "no"

        if action_name == "prepare_to_answer":
            if len(action_list) > 1 and len(action_name_set) != 1:
                tool_call_analysis += f"The Planner has generated multiple tools at the same time, including the prepare_to_answer tool. The prepare_to_answer tool cannot be invoked concurrently with other tools and needs to be called serially afterwards. An error has occurred and needs to be checked!!!!\n\n"
                analysis["correct"] = "no"
                continue

            answer_type = action_arguments["answer_type"]
            if answer_type not in ["tool", "chat"]:
                parameter_input_analysis += "The planner generated incorrect parameters for the prepare_to_answer tool; they are not one of either 'tool' or 'chat'. This is a generation error that needs to be checked immediately!!!!\n\n"
                analysis["correct"] = "no"

            if tool_flag and answer_type == "chat":
                parameter_input_analysis += "The previous message is the result of the tool's execution. However, the Planner generated the parameter for the prepare_to_answer tool as 'chat', which is illegal and causes an error. This needs to be checked!!!! In this case, the parameter for the prepare_to_answer tool can only be the name of a tool from the [Tool List], or 'tool'.\n\n"
                analysis["correct"] = "no"

    fetch_data = None
    if enable_llm:
        if len(action_list) > 1 or "serial" in thought or "parallel" in thought or "serial" in plan or "parallel" in plan:
            llm_parallel_correct, llm_parallel_analysis, llm_parallel_checker_result, fetch_data = llm_parallel_checker(messages, tools, env_info, request_func)
            if llm_parallel_correct == "no":
                analysis["correct"] = llm_parallel_correct
                tool_call_analysis += llm_parallel_analysis["tool_call_analysis"]

            pattern = r"Serially invoke the \w+ tool for ([一二三四五六七八九十]+) times."
            matches = re.findall(pattern, plan)
            for match in matches:
                analysis["correct"] = "no"
                tool_name = match[0]
                call_times = match[1]
                tool_call_analysis += f"The plan provided by Planner indicates that the {tool_name} tool should be called serially {call_times} times, but the number of tools listed in the Action_List is greater than 1, which is contradictory and needs to be checked!!!!!!"

    analysis["tool_call_analysis"] = tool_call_analysis
    analysis["parameter_input_analysis"] = parameter_input_analysis

    rule_checker_result = f"Checker_Planner：\n```json\n{json.dumps(analysis, ensure_ascii=False, indent=4)}\n```"
    logger.info(f"rule_checker\n{rule_checker_result}\n")
    return analysis["correct"], analysis, rule_checker_result, fetch_data


def llm_parallel_checker(messages, tools, env_info, request_func):
    all_tool_name, all_tool_required_info = get_all_tool_info(tools)
    language = os.getenv("language")
    if language == "zh":
        checker_serial_system_prompt_template = checker_serial_system_prompt_template_zh
    else:
        checker_serial_system_prompt_template = checker_serial_system_prompt_template_en
    checker_system_prompt = checker_serial_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                                 .replace("{{{env_info}}}", env_info) \
                                                                 .replace("{{{all_tool_required_info}}}", all_tool_required_info)
    messages_new = [
        {
            "role": "system",
            "content": checker_system_prompt
        }
    ]
    # print(checker_system_prompt)
    messages_new.extend(messages)
    while True:
        res = request_func(messages_new)
        logger.info(f"llm_parallel_checker: {res}\n")
        if "ask_user_for_required_parameters" not in res:
            break
    res_obj = parse_answer(res)
    correct = res_obj["correct"]
    fetch_data = {"task": "llm_parallel_checker", "tools": tools, "env_info": env_info, "messages": messages_new, "answer": res}
    return correct, res_obj, res, fetch_data


def checker_planner(messages, tools, env_info, tool_flag, request_func, enable_llm=True):
    language = os.getenv("LANGUAGE")
    if language == "zh":
        rule_correct, rule_analysis, rule_checker_result, fetch_data = rule_checker_zh(messages, tools, env_info, tool_flag, request_func, enable_llm)
    else:
        rule_correct, rule_analysis, rule_checker_result, fetch_data = rule_checker_en(messages, tools, env_info, tool_flag, request_func, enable_llm)
    correct = rule_correct
    checker_result = f"Checker_Planner：\n```json\n{json.dumps(rule_analysis, ensure_ascii=False, indent=4)}\n```"
    return correct, checker_result, fetch_data
