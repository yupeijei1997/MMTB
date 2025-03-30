import json
import os
import copy
import argparse
import sys
current_path_list = os.getcwd().split("/")[:-2]
current_path = "/".join(current_path_list)
print(f"current_path: {current_path}\n")
sys.path.append(current_path)

from utils import read_file_to_json,get_random_pathname
from tool_call_graph import eval_by_tool_call_graph
from handle.handles import tool_handle_map
from tqdm import tqdm


def str2bool(v):
    '''
    Transform string to bool.

    Arguments:
        v (str): The value to be converted.

    Returns:
        bool: The converted value.
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_messages_until_task(messages, task_id, task, history, is_english, remove_role=True):
    '''
    整合历史消息并根据任务ID和任务内容过滤消息，同时根据语言移除角色标识。

    Arguments:
        messages (list): 包含消息记录的列表，每条记录是一个字典，包含内容和角色等信息。
        task_id (int): 任务ID，用于定位特定任务的消息。
        task (str): 任务内容，用于验证消息中是否包含该任务。
        history (list): 历史消息列表，每个元素是一个消息列表。
        is_english (bool): 是否为英文消息，用于确定如何移除角色标识。
        remove_role (bool): 是否移除消息中的角色标识，默认为True。

    Returns:
        list: 整合后的消息列表，根据任务ID和任务内容过滤，并移除了角色标识。
    '''
    new_messages = []
    try:
        for history_messages in history:
            new_messages += history_messages
        assert len(new_messages) % 2 == 0
        assert task in messages[task_id]["content"]
        new_messages += messages[:task_id+1]
        assert len(new_messages) % 2 == 1
        role = "user"
        for m in new_messages:
            assert m["role"] == role
            role = "assistant" if role == "user" else "user"
            if not remove_role:
                continue
            if is_english:
                colon_idx = m["content"].find(":")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx].lower() in [
                        "ai", "ai agent", "user", "ai agent assistant"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
            else:
                colon_idx = m["content"].find("：")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx] in [
                        "用户", "AI Agent助手", "AI Agent"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
    except Exception as e:
        # ipdb.set_trace()
        print(f"error: {e}")

    return new_messages


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--new_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt4o")
    parser.add_argument("--data_path", type=str, default="./data/Multi-Mission-Tool-Bench.jsonl")
    parser.add_argument("--output_path", type=str, default="./result")
    parser.add_argument("--model_url", type=str, default="http://111.111.111.111:12345")
    parser.add_argument("--continue_file", type=str, default=None)
    parser.add_argument("--remove_role", type=str2bool, default=True)
    parser.add_argument("--contain_context", type=str2bool, default=True)
    parser.add_argument("--debug_id", type=str, default=None)
    parser.add_argument("--debug_idx", type=str, default=None)
    parser.add_argument("--skip_num", type=int, default=0)
    parser.add_argument("--retry_num", type=int, default=1)
    args = parser.parse_args()
    return args


def add_args_info_into_filename(args):
    args_dict = vars(args)
    key = []
    if args.new_key is not None:
        key.append(args.new_key)
    key.append(args_dict["model"])
    key.append(args_dict["language"])
    for k in args_dict:
        if type(args_dict[k]) == bool:
            key.append(k)
    return "_".join(key)


def split_messages_by_equal(messages):
    messages_list = []
    now_message = []
    for m in messages:
        if type(m) == str and "=====" in m:
            messages_list.append(copy.copy(now_message))
            now_message = []
        else:
            now_message.append(m)
    if len(now_message) != 0:
        messages_list.append(now_message)
    return messages_list


def main(args):
    data = read_file_to_json(args.data_path)

    res_data = []
    path_ = get_random_pathname(args.output_path, "jsonl", keys = add_args_info_into_filename(args), need_time=True)
    is_english = False if args.language == "zh" else True
    error_list = []
    too_long_continue = 0
    task_length = 0
    process_cnt = 0
    debug_mode = args.debug_id and args.debug_idx
    if args.continue_file and "None" not in args.continue_file and os.path.exists(args.continue_file) and args.model in args.continue_file:
        continue_file = read_file_to_json(args.continue_file)
        res_data = continue_file
        path_ = args.continue_file.replace(".unfinish", "")
        task_length += len(res_data)
        print(f"continue file: {args.continue_file}")
        print(f"task_length: {task_length}")
    elif args.skip_num != 0:
        data = data[int(args.skip_num /4)+1:]

    for item in tqdm(data):
        try:
            if debug_mode and args.debug_id not in item["id"]:
                continue
            task_list = item["english_task"] if is_english else item["task"]
            answer_lists = item["english_answer_list"] if is_english else item["answer_list"]
            messages_list = item["english_messages"] if is_english else item["messages"]
            tools_list = item["english_tools"] if is_english else item["tools"]
            messages_list = split_messages_by_equal(messages_list)
            assert type(task_list) == list and type(answer_lists[0]) == list
            assert len(task_list) == len(answer_lists) and len(task_list) == len(messages_list)
            if not args.contain_context and len(task_list) == 1:
                continue
            if type(item["env_info"]) == str:
                item["env_info"] = [item["env_info"] for _ in range(len(task_list))]
            item["env_info"] = [
                env_info[:env_info.find("星期")].strip()
                for env_info in item["env_info"] if "星期" in env_info
            ]
            for id_, task_id, task, answer_list, messages, env_info in zip(
                range(len(task_list)), item["task_ids"],
                task_list, answer_lists, messages_list, item["env_info"]
            ):
                if debug_mode and int(args.debug_idx) != id_:
                    continue
                if not args.contain_context and id_ == 0:
                    continue
                process_cnt += 1
                if process_cnt <= len(res_data):
                    continue
                simulator, response_continue = tool_handle_map[args.model]
                simulator = simulator(args.model_url, is_english)
                if args.contain_context:
                    messages = get_messages_until_task(
                        messages, task_id, task, messages_list[:id_], is_english, args.remove_role
                    )
                else:
                    messages = get_messages_until_task(
                        messages, task_id, task, [], is_english, args.remove_role
                    )
                messages_length = len(messages)
                predict_label, predict_is_optimal, predict_result, answer_result = eval_by_tool_call_graph(
                    simulator.request_funcall,
                    messages,
                    tools_list,
                    answer_list,
                    response_continue,
                    env_info=env_info,
                    retry_num=args.retry_num
                )
                res_data.append({
                    "id": item["id"],
                    "idx": id_,
                    "messages": messages,
                    "messages_length": messages_length,
                    "task_id": task_id,
                    "type": item["type"],
                    "tools": tools_list,
                    "task": task,
                    "answer_list": answer_list,
                    "predict_result": predict_result,
                    "predict_label": predict_label,
                    "predict_is_optimal": str(predict_is_optimal),
                    "answer_result": answer_result,
                    "turn_type": [(_ if type(_) == bool else _ == "真") for _ in item.get("turn_type", [])],
                    "turn_subtypes":  item.get("turn_subtypes", []),
                })
                if len(res_data) % 10 == 1:
                    print(task)
                    print(predict_result)
                with open(path_ + ".unfinish", "w", encoding="utf-8") as f:
                    for res in res_data:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"error: {e}")
            error_list.append([item["id"], e])
            # with open("error.jsonl", "w", encoding="utf-8") as f:
            #     for error in error_list:
            #         f.write(json.dumps(error, ensure_ascii=False) + "\n")

        task_length += len(item["task"])
        # print(f"{task_length}:-:{len(res_data)}:-:{process_cnt}")

    print(f"error cnt: {len(error_list)}")
    print(f"too long: {too_long_continue}")
    if not debug_mode:
        os.system(f'mv {path_}.unfinish {path_}')


if __name__ == "__main__":
    args = parse_argument()
    main(args)
    # add_args_info_into_filename(args)