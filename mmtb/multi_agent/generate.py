import copy
import random
import json
import os
import argparse

from handle.handles import agent_handle_map
from agent import user_single_tool, user_multi_tool, user_multi_tool_parallel, user_multi_tool_serial_parallel, \
                  user_ask, user_answer_ask, user_vague_answer_ask, user_chat, user_continue_question, \
                  agent_ask, agent_answer, agent_answer_chat, \
                  planner, tool, \
                  checker_planner, checker_tool
from utils import read_json_file_to_list, get_random_date, parse_answer, transform_train_data, logger, \
                  ask_user_for_help_tool, prepare_to_answer_tool
from datetime import datetime


def pipeline(node_list, messages, tools, env_info, fetch_data_list, layer_num_total, agent_handle_to_model):
    FAILED = False
    dialog_turns = 0
    tools.append(ask_user_for_help_tool)
    tools.append(prepare_to_answer_tool)
    all_tool_name = [_["function"]["name"] for _ in tools]
    all_tool_name.append("ask_user_for_required_parameters")
    all_tool_name.append("prepare_to_answer")

    def one_turn_pipeline():
        nonlocal agent_handle_to_model
        language = os.getenv("LANGUAGE")
        user_handle = agent_handle_to_model["user"]
        planner_handle = agent_handle_to_model["planner"]
        tool_handle = agent_handle_to_model["tool"]
        agent_handle = agent_handle_to_model["agent"]
        checker_handle = agent_handle_to_model["checker"]
        END = False
        turns = 0
        nonlocal messages
        nonlocal FAILED

        while True:
            if messages[-1]["content"].startswith("用户") or messages[-1]["content"].startswith("User") or "Tool：\n```json" in messages[-1]["content"] or \
                    "Tool:\n```json" in messages[-1]["content"] or messages[-1]["content"].startswith("Checker_Tool"):
                tool_flag = "Tool：\n```json" in messages[-1]["content"]
                checker_flag = False
                for _ in range(3):
                    if language == "zh":
                        messages.append({"role": "user", "content": "切换角色为Planner，继续输出Planner的决策，注意：1、你每次生成时，如果你之前生成了错误的决策，本轮请不要将以前生成错误的结果或是计划调整在Thought和Plan里说明，而是当成全新的一轮给出Thought和Plan。\n2、务必注意不要Plan中提到调用prepare_to_answer工具和调用ask_user_for_required_parameters工具。"})
                    else:
                        messages.append({"role": "user", "content": "Switch to the role to Planner and continue to output the Planner's decisions. Note: 1. Each time you generate, if you have previously generated incorrect decisions, please do not explain the previously generated incorrect results or plan adjustments in the Thought and Plan sections for this round. Instead, treat it as a brand new round and provide your Thought and Plan. \n2. Be sure not to mention the use of the prepare_to_answer tool and the ask_user_for_required_parameters tool in the Plan."})
                    res, fetch_data_planner = planner(messages, tools, env_info, planner_handle.request_model)
                    planner_part_ = [
                        {"role": "assistant", "content": res}
                    ]
                    messages.extend(planner_part_)

                    if language == "zh":
                        messages.append({"role": "user", "content": "切换角色为Checker，继续输出Checker的检查结果"})
                    else:
                        messages.append({"role": "user", "content": "Switch the role to Checker and continue to output the Checker's inspection results."})
                    correct, res, fetch_data_checker = checker_planner(messages, tools, env_info, tool_flag, checker_handle.request_model, True)
                    fetch_data_list.append(fetch_data_checker)
                    if correct == "yes":
                        messages = messages[:-1]
                        fetch_data_list.append(fetch_data_planner)
                        checker_flag = True
                        break

                    checker_planner_part_ = [
                        {"role": "assistant", "content": res}
                    ]
                    messages.extend(checker_planner_part_)

                if not checker_flag:
                    FAILED = True
                    break

            elif messages[-1]["content"].startswith("Agent"):
                if language == "zh":
                    messages.append({"role": "user", "content": "切换角色为用户，继续输出用户的回复"})
                else:
                    messages.append({"role": "user", "content": "Switch the role to User and continue to output the User's responses."})
                if random.choice([True, True, True, False]):
                    res, fetch_data = user_answer_ask(messages, tools, env_info, user_handle.request_model)
                else:
                    res, fetch_data = user_vague_answer_ask(messages, tools, env_info, user_handle.request_model)
                part_ = [
                    {"role": "assistant", "content": res}
                ]
                messages.extend(part_)
                fetch_data_list.append(fetch_data)

            elif messages[-1]["content"].startswith("Planner") or messages[-1]["content"].startswith("Checker_Planner"):

                if messages[-1]["content"].startswith("Checker_Planner"):
                    parse_content = parse_answer(messages[-3]["content"])
                else:
                    parse_content = parse_answer(messages[-1]["content"])

                action_list = parse_content["Action_List"]
                assert isinstance(action_list, list)
                for action in action_list:
                    assert action["name"] in all_tool_name

                if action_list[0]["name"] == "ask_user_for_required_parameters":
                    if language == "zh":
                        messages.append({"role": "user", "content": "切换角色为Agent助手，继续输出ask_user_for_required_parameters的询问信息"})
                    else:
                        messages.append({"role": "user", "content": "Switch the role to Agent and continue to output the inquiry information."})
                    ask_res, fetch_data = agent_ask(messages, tools, env_info, agent_handle.request_model)
                    part_ = [
                        {"role": "assistant", "content": ask_res}
                    ]
                    messages.extend(part_)
                    fetch_data_list.append(fetch_data)

                elif action_list[0]["name"] == "prepare_to_answer":
                    if action_list[0]["arguments"]["answer_type"] == "tool":
                        if language == "zh":
                            messages.append({"role": "user", "content": "切换角色为Agent助手，继续输出prepare_to_answer的总结回复，注意不要输出```markdown```这种字样"})
                        else:
                            messages.append({"role": "user", "content": "Switch the role to Agent and continue to output summary replies, be careful not to output words like ```markdown```."})
                        answer_res, fetch_data = agent_answer(messages, tools, env_info, agent_handle.request_model)
                    else:
                        if language == "zh":
                            messages.append({"role": "user", "content": "切换角色为Agent助手，继续输出prepare_to_answer的直接回复"})
                        else:
                            messages.append({"role": "user", "content": "Switch the role to Agent and continue to output direct replies."})
                        answer_res, fetch_data = agent_answer_chat(messages, tools, env_info, agent_handle.request_model)

                    part_ = [
                        {"role": "assistant", "content": answer_res}
                    ]
                    messages.extend(part_)
                    fetch_data_list.append(fetch_data)
                    END = True

                else:
                    checker_flag = False
                    for _ in range(3):
                        if language == "zh":
                            messages.append({"role": "user", "content": "切换角色为Tool，继续输出Tool的执行结果"})
                        else:
                            messages.append({"role": "user", "content": "Switch the role to Tool and continue to output the execution results of Tool."})
                        res, fetch_data = tool(messages, tools, env_info, tool_handle.request_model)
                        tool_part_ = [
                            {"role": "user", "content": res}
                        ]
                        messages.extend(tool_part_)
                        if language == "zh":
                            messages.append({"role": "user", "content": "切换角色为Checker，继续输出Checker的检查结果"})
                        else:
                            messages.append({"role": "user", "content": "Switch the role to Checker and continue to output the Checker's inspection results."})

                        correct, res = checker_tool(messages, action_list, tools, env_info, checker_handle.request_model)
                        if correct == "yes":
                            messages = messages[:-1]
                            fetch_data_list.append(fetch_data)
                            checker_flag = True
                            break

                        checker_planner_part_ = [
                            {"role": "assistant", "content": res}
                        ]
                        messages.extend(checker_planner_part_)

                    if not checker_flag:
                        FAILED = True
                        break

            else:
                break

            turns += 1
            if END or turns >= 100:
                return

    while True:
        one_turn_pipeline()

        if FAILED:
            break

        dialog_turns += 1
        if dialog_turns >= layer_num_total:
            break

        language = os.getenv("LANGUAGE")
        if language == "zh":
            messages.append({"role": "user", "content": "切换角色为用户，继续提出新任务"})
        else:
            messages.append({"role": "user", "content": "Switch the role to user and continue to propose new tasks."})
        user_handle = agent_handle_to_model["user"]
        res, fetch_data = user_continue_question(messages, tools, env_info, user_handle.request_model, node_list[dialog_turns])
        part_ = [
            {"role": "assistant", "content": res}
        ]
        messages.extend(part_)
        fetch_data_list.append(fetch_data)

    return FAILED, messages


node_to_user_agent = {
    "ST": [user_single_tool],
    "MT": [user_multi_tool, user_multi_tool_parallel, user_multi_tool_serial_parallel],
    "CQ": [user_ask],
    "CC": [user_chat]
}


def gen_one_data(tools, node_list, layer_num_total, agent_handle_to_model):
    logger.info(f"gen_one_data\n\nAsk questions about the following tools：\n{json.dumps(tools, ensure_ascii=False, indent=4)}\n")

    messages = []
    fetch_data_list = []
    language = os.getenv("LANGUAGE")
    if language == "zh":
        env_info = "当前时间：" + get_random_date()
    else:
        env_info = "Current Time：" + get_random_date()
    logger.info(f"gen_one_data\n\n{env_info}\n")

    # first turn
    user_start_func = random.choice(node_to_user_agent[node_list[0]])
    messages_ret = None
    user_handle = agent_handle_to_model["user"]

    try:
        user_task_message, fetch_data = user_start_func(messages, tools, user_handle.request_model)
        messages.extend(user_task_message)
        fetch_data_list.append(fetch_data)
        FAILED, messages_ret = pipeline(node_list, messages, tools, env_info, fetch_data_list, layer_num_total, agent_handle_to_model)
    except Exception as e:
        logger.info(f"gen_one_data\nerror: {e}")
        FAILED = True

    return FAILED, messages_ret, tools, env_info, fetch_data_list


def main(all_path_list, layer_num_total, agent_handle_to_model):
    language = os.getenv("LANGUAGE")
    if language == "zh":
        tools_all_list = read_json_file_to_list("tools/mmtb_tools_zh.jsonl")
    else:
        tools_all_list = read_json_file_to_list("tools/mmtb_tools_en.jsonl")

    current_time = datetime.now()  # 打印当前日期和时间
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    for current_layer_path_list in all_path_list[layer_num_total - 1]:
        fout = open(f"result/{formatted_time}_train.jsonl", "a+")
        fout_origin = open(f"result/{formatted_time}_train_origin.jsonl", "a+")
        fout_fetch_data = open(f"result/{formatted_time}_train_fetch_data.jsonl", "a+")
        logger.info(f"current_layer_path_list: {current_layer_path_list}")
        tools_select = random.choice(tools_all_list)
        FAILED, messages_ret, tools, env_info, fetch_data_list = gen_one_data(tools_select, current_layer_path_list, layer_num_total, agent_handle_to_model)
        if FAILED:
            continue

        DATA_FAILED, train_data_example, train_data_example_origin = transform_train_data(messages_ret, tools, env_info)
        fout.write(json.dumps(train_data_example, ensure_ascii=False) + "\n")
        fout_origin.write(json.dumps(train_data_example_origin, ensure_ascii=False) + "\n")

        for fetch_data in fetch_data_list:
            if fetch_data is not None:
                fout_fetch_data.write(json.dumps(fetch_data, ensure_ascii=False) + "\n")

        fout.close()
        fout_origin.close()
        fout_fetch_data.close()


def gen_path(layer_num_total):
    task_type_list = ["ST", "MT", "CQ", "CC"]
    all_path_list = [[["ST"], ["MT"], ["CQ"], ["CC"]]]
    for current_layer_num in range(1, layer_num_total):
        logger.info(f"current_layer_num: {current_layer_num}")
        last_path_list = all_path_list[current_layer_num - 1]
        logger.info(f"last_path_list: {last_path_list}")
        logger.info(f"len(last_path_list): {len(last_path_list)}")
        current_layer_path_list = []
        for path in last_path_list:
            for task_type in task_type_list:
                current_path = copy.deepcopy(path)
                current_path.append(task_type)
                current_layer_path_list.append(current_path)
        logger.info(f"current_layer_path_list: {current_layer_path_list}")
        logger.info(f"len(current_layer_path_list): {len(current_layer_path_list)}\n")
        all_path_list.append(current_layer_path_list)
    return all_path_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two lists of strings.")

    # Add arguments for two lists of strings
    parser.add_argument(
        "--layer-num-total", type=str, default=4, help="The total number of levels in the path tree"
    )
    parser.add_argument(
        "--user-model", type=str, default="gpt4o", help="Model used by the user agent"
    )
    parser.add_argument(
        "--planner-model", type=str, default="gpt4o", help="Model used by the planner agent"
    )
    parser.add_argument(
        "--tool-model", type=str, default="gpt4o", help="Model used by the tool agent"
    )
    parser.add_argument(
        "--agent-model", type=str, default="gpt4o", help="Model used by the AI agent"
    )
    parser.add_argument(
        "--checker-model", type=str, default="gpt4o", help="Model used by the checker agent"
    )
    args = parser.parse_args()

    user_model = agent_handle_map[args.user_model]
    planner_model = agent_handle_map[args.planner_model]
    tool_model = agent_handle_map[args.tool_model]
    agent_model = agent_handle_map[args.agent_model]
    checker_model = agent_handle_map[args.checker_model]

    agent_handle_to_model = {
        "user": user_model(),
        "planner": planner_model(),
        "tool": tool_model(),
        "agent": agent_model(),
        "checker": checker_model()
    }

    all_path_list = gen_path(args.layer_num_total)
    main(all_path_list, args.layer_num_total, agent_handle_to_model)
