import copy
import json
import ipdb
import jieba
import argparse
import logging
import re
import warnings
import os

from utils import read_file_to_json, write_json_to_file, write_list_of_list_to_csv
from collections import Counter
from rouge_score import rouge_scorer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDNN_LOGINFO_DBG"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.captureWarnings(True)
warnings.filterwarnings("always", category=DeprecationWarning,
                        module=r"^{0}.".format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning", category=DeprecationWarning)


def easy_dict_count(label_cnt, label):
    label_cnt[label] = label_cnt.get(label, 0) + 1


def print_dict_by_key(dict_, sort_by=None, need_percent=False):
    dict_items = []
    sum_ = 0
    for item in dict_:
        dict_items.append([item, dict_[item]])
        sum_ += dict_[item]

    if sort_by is not None:
        if sort_by == "key":
            dict_items = sorted(dict_items, key=lambda x: x[0])
        elif sort_by == "num":
            dict_items = sorted(dict_items, key=lambda x: x[1], reverse=True)
        for item in dict_items:
            print_str = "\t"
            print_str += item[0] + ":"
            print_str += "\t" + str(item[1])
            if need_percent:
                print_str += "\t" + "{:.02f}%".format(item[1] / sum_ * 100)
            print(print_str)


class _DummyTokenizer:
    def tokenize(self, text: str):
        return list(jieba.cut(text))


scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], tokenizer=_DummyTokenizer()
)


def remove_more_space(text):
    return "".join(text.split())


def check_single_arguments(param1, param2, types):
    param1 = float(param1) if type(param1) == int else param1
    param2 = float(param2) if type(param2) == int else param2
    if type(param1) != type(param2):
        return 0, ["type error"]

    if param1 == param2:
        return 1, []

    if (
            type(param1) == str
            and (
            param1.lower() == param2.lower()
            or remove_more_space(param1).lower() == remove_more_space(param2).lower()
    )
    ):
        return 1, []

    if type(param1) == type(param2) and type(param1) == dict:
        score_list = []
        error_types = []
        if "items" in types:
            item_type = types.get("items", {"type": "string"})
        elif "properties" in types:
            item_type = types.get("properties", {"type": "string"})
        else:
            item_type = types

        if "properties" in item_type:
            item_type = item_type["properties"]

        assert type(item_type) == dict

        for key in param1:
            if key not in item_type:
                return 0, ["param hallucination"]
            if key not in param2:
                arguments_description = item_type[key].get("description", "")
                if str(param1[key]).lower() in arguments_description.lower():
                    score_list.append(1)
                elif str(param1[key]).lower() == "none" and (
                        "null" in arguments_description.lower()
                ):
                    score_list.append(1)
                else:
                    score_list.append(0)
                    error_types.append("param value hallucination")
            else:
                score_, types_ = check_single_arguments(param1[key], param2[key], item_type[key])
                score_list.append(score_)
                error_types += types_

        if len(score_list) == 0:
            if len(param2) == 0:
                return 1, []
            else:
                return 0, ["param value error"]

        return sum(score_list) / len(score_list), error_types

    if types["type"] in ["integer", "boolean", "float", "number"]:
        return 0, ["param value error"]

    elif types["type"] in ["string"]:
        if (
                re.match(r"^\d+\.\d+,\d+\.\d+$", param1) and
                re.match(r"^\d+\.\d+,\d+\.\d+$", param2)
        ):
            return 1, []
        global scorer
        score = scorer.score(param1, param2)["rougeL"].fmeasure
        return score, [] if score > 0.7 else ["param value error"]

    elif types["type"] in ["array"]:
        score_list = []
        error_types = []
        if type(param1) == list:
            sorted_params1 = sorted(param1, key=lambda x: json.dumps(x, sort_keys=True))
            sorted_params2 = sorted(param2, key=lambda x: json.dumps(x, sort_keys=True))
        else:
            sorted_params1 = sorted(param1.values())
            sorted_params2 = sorted(param2.values())
        if len(sorted_params1) != len(sorted_params2):
            return 0, ["param value error"]

        item_type = types.get("items", {"type": "string"})
        if "properties" in item_type:
            item_type = item_type["properties"]
        for p1, p2 in zip(sorted_params1, sorted_params2):
            score_, types_ = check_single_arguments(p1, p2, item_type)
            score_list.append(score_)
            error_types += types_

        return sum(score_list) / len(score_list), error_types

    else:
        assert False, f"Unknown type: {types}, {param1}, {type(param1)}, {param2}, {type(param2)}"


def check_arguments(predict, answer, tool):
    assert "arguments" in predict, f"Predict miss arguments, get {predict.keys()}"
    if type(predict["arguments"]) == str:
        try:
            predict["arguments"] = json.loads(predict["arguments"])
        except:
            raise f"Json load arguments error: {predict['arguments']}"

    tool_parameters_dict = tool["function"]["parameters"]
    required_parameters = tool_parameters_dict.get("required", [])
    for parameter in required_parameters:
        if parameter not in predict["arguments"]:
            return 0, {parameter: 0}, ["miss required params"]

    if len(predict["arguments"]) == 0:
        if len(answer["arguments"]) == 0:
            return 1, {}, []

    arguments_score = []
    arguments_score_dict = {}
    error_type = []
    tool_parameters_dict = tool_parameters_dict["properties"]
    for arguments in predict["arguments"]:
        if arguments not in tool_parameters_dict:
            arguments_score.append(0)
            arguments_score_dict[arguments] = 0
            error_type.append("param hallucination")
            continue
        value = predict["arguments"][arguments]
        if arguments in answer["arguments"]:
            score, type_ = check_single_arguments(
                value, answer["arguments"][arguments],
                tool_parameters_dict[arguments]
            )
            arguments_score.append(score)
            error_type += type_
        else:
            arguments_description = tool_parameters_dict[arguments].get("description", "")
            if str(value).lower() in arguments_description.lower():
                arguments_score.append(1)
            elif str(value).lower() == "none" and (
                    "null" in arguments_description.lower()
            ):
                arguments_score.append(1)
            else:
                arguments_score.append(0)
                error_type.append("param value hallucination")
        arguments_score_dict[arguments] = arguments_score[-1]

    return sum(arguments_score) / len(arguments_score), arguments_score_dict, error_type


def check_every_function_arguments(answer_list, predict_result, tools, type_="flag", item_id=None):
    assert type_ in ["flag", "score"]
    tools2dict = {_["function"]["name"]: _ for _ in tools}
    answer_by_name = {}
    for answer in answer_list:
        if answer["action"]["name"] in ["prepare_to_answer", "ask_user_for_required_parameters"]:
            continue
        if answer["action"]["name"] not in answer_by_name:
            answer_by_name[answer["action"]["name"]] = []
        answer_by_name[answer["action"]["name"]].append(answer)

    predict_by_name = {}
    toolcall2idx = {}
    for idx, predict_step in enumerate(predict_result):
        if predict_step.get("tool_calls", None) is None:
            continue
        for tool_call in predict_step["tool_calls"]:
            tool_call = {
                "idx": idx,
                "action": tool_call["function"]
            }
            if tool_call["action"]["name"] not in predict_by_name:
                predict_by_name[tool_call["action"]["name"]] = []
            predict_by_name[tool_call["action"]["name"]].append(tool_call)
            toolcall2idx[json.dumps(tool_call)] = idx
    assert len(predict_by_name) == len(answer_by_name)

    function_score = {}
    bad_function_pair = {"items": [], "tool": []}
    all_error_type = []

    if item_id is not None:
        bad_function_pair["item_id"] = item_id

    for function_name in list(predict_by_name.keys()):
        assert len(predict_by_name[function_name]) == len(answer_by_name[function_name])
        predict_list = predict_by_name[function_name]
        answer_list = answer_by_name[function_name]
        already_paired_answer = []
        function_score[function_name] = []
        for predict in predict_list:
            idx_ = toolcall2idx[json.dumps(predict)]
            paired_score = -1
            paired_id = -1
            for id_, answer in enumerate(answer_list):
                if id_ in already_paired_answer:
                    continue
                score_, score_dict, error_type = check_arguments(
                    predict["action"],
                    answer["action"],
                    tools2dict[function_name]
                )
                if score_ > paired_score:
                    paired_score = score_
                    paired_id = id_

            if paired_score < 0.7:
                bad_item = {
                    "predict": predict["action"],
                    "answer": answer["action"],
                    "score_dict": score_dict
                }
                bad_function_pair["items"].append(bad_item)
                if tools2dict[function_name]["function"]["name"] not in [_["function"]["name"] for _ in
                                                                         bad_function_pair["tool"]]:
                    bad_function_pair["tool"].append(tools2dict[function_name])
                all_error_type += [{"idx": idx_, "error_type": error_type}]
            function_score[function_name].append(paired_score)
            already_paired_answer.append(paired_id)

    if type_ == "flag":
        for function_name in function_score:
            for score in function_score[function_name]:
                if score < 0.7:
                    return False, bad_function_pair, all_error_type
        return True, bad_function_pair, all_error_type
    else:
        return sum([sum(_) for _ in function_score.values()]) / sum(
            [len(_) for _ in function_score.values()]), bad_function_pair, all_error_type


def param_error_type_analysis(data):
    error_param_type = {}

    def get_types(properties):
        type_list = []
        if "type" not in properties:
            return type_list
        type_list += [properties["type"]]
        if "properties" in properties:
            type_list.append([get_types(properties["properties"][_]) for _ in properties["properties"]])
        if "items" in properties:
            type_list.append(get_types(properties["items"]))

        return type_list

    cnt = 0
    for item in data:
        for item_dict in item["items"]:
            function_name = item_dict["answer"]["name"]
            tool_info = \
            [_ for _ in item["tool"] if _["function"]["name"] == function_name][0]["function"]["parameters"][
                "properties"]
            for key, value in item_dict["score_dict"].items():
                if value >= 0.7:
                    continue
                param_type = json.dumps(get_types(tool_info.get(key, {})))
                # if param_type == "[]":
                #     print(key, item_dict["score_dict"], tool_info)
                error_param_type[param_type] = error_param_type.get(param_type, 0) + 1
                cnt += 1

    print("\nerror param type:")
    print("=" * 60)
    for k, v in sorted(error_param_type.items(), key=lambda x: -x[1]):
        print(k, v, v / cnt)


def draw_matrix_by_type_and_index(matrix, answer_depth=4):
    index_matrix = {}
    type_matrix = {}
    for index in matrix:
        for type_ in matrix[index]:
            if type_ not in type_matrix:
                type_matrix[type_] = [0, 0]
            type_matrix[type_][0] += matrix[index][type_][0]
            type_matrix[type_][1] += matrix[index][type_][1]
            if index not in index_matrix:
                index_matrix[index] = [0, 0]
            index_matrix[index][0] += matrix[index][type_][0]
            index_matrix[index][1] += matrix[index][type_][1]

    # draw, x axis is type, y axis is index, value is correct rate
    x_axis = ["单", "多", "拒", "反"] + ["all", "num"]
    y_axis = list(range(answer_depth)) + ["all"]
    res_axis = []
    for type_ in x_axis:
        res_axis.append([type_])
        for index in y_axis:
            if type_ == "num":
                if index == "all":
                    res = sum([index_matrix[_][1] for _ in index_matrix])
                else:
                    res = index_matrix[index][1]
                res_axis[-1].append(str(res))
                continue
            if type_ == "all" and index == "all":
                res = sum([_[0] for _ in index_matrix.values()]) / sum([_[1] for _ in index_matrix.values()])
                res2 = sum([_[0] for _ in type_matrix.values()]) / sum([_[1] for _ in type_matrix.values()])
                assert res == res2
                # print("all", res)
                res_axis[-1].append(str(res)[:8])
            elif type_ == "all":
                res = index_matrix[index][0] / index_matrix[index][1]
                # print(index, res)
                res_axis[-1].append(str(res)[:8])
            elif index == "all":
                res = type_matrix[type_][0] / type_matrix[type_][1]
                # print(type_, res)
                res_axis[-1].append(str(res)[:8])
            else:
                res = matrix[index].get(type_, [0, 1e-5])[0] / matrix[index].get(type_, [0, 1e-5])[1]
                # print(index, type_, res)
                res_axis[-1].append(str(res)[:8])

    # transpose res axis and print
    res_axis = list(map(list, zip(*res_axis)))
    print("Matrix:")
    print("\n".join([" ".join(_) for _ in res_axis]))
    # 总、单、多、拒、反、0、1、2、3
    print("\nresult：")
    print("all\t单\t多\t拒\t反\t" + "\t".join([str(_) for _ in range(answer_depth)]))
    tmp_res = [res_axis[-1][-2], res_axis[-1][0], res_axis[-1][1], res_axis[-1][2], res_axis[-1][3]]
    for i in range(answer_depth):
        tmp_res.append(res_axis[i + 1][-2])
    print(" ".join(tmp_res) + "\n")
    return tmp_res


def weight_matrix_auto(matrix, index, type, label, more_weight=None):
    if index not in matrix:
        matrix[index] = {}
    if type[index] not in matrix[index]:
        matrix[index][type[index]] = [0, 0]
    matrix[index][type[index]][1] += 1 + 0 if more_weight is None else more_weight
    if label == "correct":
        matrix[index][type[index]][0] += 1 + 0 if more_weight is None else more_weight
    return matrix


def matrix_calculate_variance(matrix):
    route_matrix = {}
    for k in matrix:
        if len(k) == 1:
            continue
        route_matrix[k[:-1]] = route_matrix.get(k[:-1], []) + matrix[k]
    route_matrix = {
        k: sum(route_matrix[k]) / len(route_matrix[k])
        for k in route_matrix
    }

    route_length_matrix = {}
    for k in route_matrix:
        if len(k) not in route_length_matrix:
            route_length_matrix[len(k)] = []
        route_length_matrix[len(k)].append(route_matrix[k])
    sorted_variance = {1: 0, 2: 0, 3: 0}

    print("\nLayer variance:")
    for k, list_of_number in route_length_matrix.items():
        if len(list_of_number) <= 1:
            sorted_variance[k] = 0
        mean = sum(list_of_number) / len(list_of_number)
        variance = sum([(_ - mean) ** 2 for _ in list_of_number]) / (len(list_of_number) - 1 + 1e-5)
        sorted_variance[k] = variance
    print(" ".join([str(sorted_variance[_]) for _ in sorted(sorted_variance.keys(), reverse=True)]))


def generate_route(depth, already_list, item_list=None):
    if depth == 0:
        return []

    if item_list is None or len(item_list) == 0:
        item_list = ["单", "多", "反", "拒"]

    ret_list = []
    for item in item_list:
        ret_list.append(already_list + [item])
        ret_list += generate_route(depth - 1, already_list + [item], item_list)

    return ret_list


def matrix_calculate_triangle_data(matrix_map):
    route_matrix = {}
    for k in matrix_map:
        route_matrix[k] = route_matrix.get(k, []) + matrix_map[k]
    route_matrix = {
        k: sum(route_matrix[k]) / len(route_matrix[k])
        for k in route_matrix
    }
    list_keys = ["".join(_) for _ in generate_route(4, [])]
    depth_list = [[], [], [], []]

    for k in list_keys:
        depth_list[len(k) - 1].append(route_matrix.get(k, 0))

    return depth_list


def analysis_answer_type(answer_list):
    # return combine, serial, cs
    step_idx = []
    now_idx = []
    for item in answer_list:
        dep_f = False
        if item["action"]["name"] in ["ask_user_for_required_parameters", "prepare_to_answer"]:
            continue
        for i_ in item["dependency_list"]:
            if i_ in now_idx:
                dep_f = True
        if not dep_f:
            now_idx.append(item["idx"])
        else:
            step_idx.append(now_idx.copy())
            now_idx = [item["idx"]]

    if len(now_idx) != 0:
        step_idx.append(now_idx)
    serial_flag = False

    if len(step_idx) > 1:
        serial_flag = True
    combine_flag = False

    if any([len(_) > 1 for _ in step_idx]):
        combine_flag = True

    if serial_flag and combine_flag:
        return "cs"
    elif serial_flag:
        return "serial"
    elif combine_flag:
        return "combine"
    else:
        return None


def calculate_predict_result_steps(label, predict_results, answer_list, bad_arguments):
    if label == "correct":
        return len(answer_list)
    else:
        predict_step = 0
        for predict_result in predict_results[:-1]:
            if "tool_calls" in predict_result and predict_result["tool_calls"] is not None and len(
                    predict_result["tool_calls"]) != 0:
                predict_step += len(predict_result["tool_calls"])
            else:
                predict_step += 1

        if len(bad_arguments.get("items", [])) != 0:
            predict_step += len(predict_results[-1]["tool_calls"]) if "tool_calls" in predict_results[-1] and \
                                                                      predict_results[-1][
                                                                          "tool_calls"] is not None and len(
                predict_results[-1]["tool_calls"]) != 0 else 1
            predict_step - len(bad_arguments["items"])

    return predict_step


def compare_lists(A, B):
    # 使用Counter统计每个列表中字符串的出现次数
    count_A = Counter(A)
    count_B = Counter(B)

    # 找出A中多出或数量不同的元素
    diff_A = []
    for item, count in count_A.items():
        if item not in count_B or count > count_B[item]:
            diff_A.extend([item] * (count - count_B.get(item, 0)))

    # 找出B中多出或数量不同的元素
    diff_B = []
    for item, count in count_B.items():
        if item not in count_A or count > count_A[item]:
            diff_B.extend([item] * (count - count_A.get(item, 0)))

    return diff_A, diff_B


def check_error_type(item, all_error_type):
    def check_toolcalls(item_dict):
        return "tool_calls" in item_dict and item_dict["tool_calls"] is not None and len(
            item_dict["tool_calls"]) != 0 and type(item_dict["tool_calls"]) == list and "function" in \
               item_dict["tool_calls"][0] and "name" in item_dict["tool_calls"][0]["function"]

    type_ = None
    tool_info = {_["function"]["name"] for _ in item["tools"]}
    if item["predict_label"] == "error":
        # action error
        already_tool_list = []
        if len(item["predict_result"]) == 1 and not check_toolcalls(item["predict_result"][0]):
            type_ = "reject tool calling"
        elif len(item["answer_list"]) == 1 and item["answer_list"][0]["action"]["name"] == "prepare_to_answer":
            type_ = "call function without reject"
        else:
            for pre_res in item["predict_result"]:
                if check_toolcalls(pre_res):
                    for tc in pre_res["tool_calls"]:
                        try:
                            if tc["function"]["name"] == "filter_by_prompt":
                                type_ = "reject tool calling"
                                break
                            elif tc["function"]["name"] not in tool_info:
                                type_ = "function name hallucination"
                                break
                            else:
                                already_tool_list.append(tc["function"]["name"])
                        except Exception as e:
                            print(f"error: {e}")
                            ipdb.set_trace()
                else:
                    already_tool_list.append("ask_user_for_required_parameters")

            if type_ is None:
                answer_list = [_["action"]["name"] for _ in item["answer_list"]]
                if "ask_user_for_required_parameters" in answer_list and "ask_user_for_required_parameters" not in already_tool_list:
                    return "missing information in tool calling"
                elif len(already_tool_list) > len(item["answer_list"]) - 1:
                    type_ = "redundant tool calling"
                else:
                    diff_answer, diff_toolcall = compare_lists(answer_list, already_tool_list)
                    if "ask_user_for_required_parameters" in diff_answer:
                        type_ = "missing information in tool calling"
                    elif len(diff_toolcall) != 0:
                        for name in already_tool_list:
                            if name not in answer_list:
                                type_ = "call the wrong function"
                        if type_ is None:
                            type_ = "redundant tool calling"
                    else:
                        type_ = "early stop"
    else:
        # all_error_type: List[Dict[str, str]], item: {"idx": idx_, "error_type": error_type}
        sorted_all_error_type = sorted(all_error_type, key=lambda x: x["idx"])
        assert len(sorted_all_error_type) != 0
        error_type = sorted_all_error_type[0]["error_type"]
        # miss required params
        # param hallucination
        # param value hallucination
        # param value error
        if "miss required params" in error_type:
            type_ = "miss required params"
        elif "param hallucination" in error_type:
            type_ = "param hallucination"
        elif "type error" in error_type:
            type_ = "type error"
        elif "param value hallucination" in error_type:
            type_ = "param value hallucination"
        elif "param value error" in error_type:
            type_ = "param value error"

    try:
        assert type_ is not None
    except Exception as e:
        print(f"error: {e}")
        ipdb.set_trace()

    return type_


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


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--parameters_eval", type=str2bool, default=True)
    parser.add_argument("--weight_eval", type=str2bool, default=False)
    parser.add_argument("--skip_first", type=str2bool, default=False)
    parser.add_argument("--badpair_path", type=str, default=None)
    parser.add_argument("--debug_id", type=str, default=None)
    parser.add_argument("--debug_idx", type=int, default=None)
    parser.add_argument("--show_triangle", type=str2bool, default=False)
    parser.add_argument("--output_csv_flag", type=str2bool, default=False)
    parser.add_argument("--output_csv_path", type=str, default="./data_with_details.csv")
    parser.add_argument("--skip_none_answer", type=str2bool, default=False)
    parser.add_argument("--answer_depth", type=int, default=4)
    args = parser.parse_args()

    if "2025-02-11-10:28:34_61612c_deepseekr1_en_remove_role_contain_context.jsonl" in args.data_file or \
        "2025-02-11-11:49:57_b6a245_deepseekv3_en_remove_role_contain_context.jsonl" in args.data_file:
        args.skip_none_answer = True

    return args


def main2(args):
    data2 = read_file_to_json(args.data_file)
    time_out_cnt = 0
    task_count = {}
    task_type = {}
    rm_dup_task = {}
    dump_task = []
    task_type_count = {}
    matrix_map = {
        "num": {},
        "weight": {},
        "params": {},
        "messages": {},
        "best_optical_rate": {"sum": [0, 0], "combine": [0, 0], "cs": [0, 0]},  # correct, sum, combine, cs
        "serial_complete_rate": {"sum": [0, 0], "serial": [0, 0], "cs": [0, 0]},
        "combine_tools_rate": [0, 0],
        "serial_tools_rate": [0, 0],
        "cs_tools_rate": [0, 0],
        "route_matrix": {},
        "error_type": {},
        "error_param_type": {},
        "turn_type_rate": {"True": [0, 0], "False": [0, 0]},
        "turn_subtype_rate": {"指代理解": [0, 0], "省略成分": [0, 0], "长期记忆": [0, 0]},
    }
    error_arguments = 0
    bad_function_pairs = []

    for item in data2:
        if args.skip_first and len(item["type"]) == 1:
            continue
        id_ = (
            item["id"] if type(item["id"]) == str
            else "".join(item["id"])
        )
        idx_ = item["idx"]
        if args.debug_id is not None and args.debug_id in id_ and args.debug_idx == item["idx"]:
            ipdb.set_trace()
            print("here")

        task = item["task"]
        if f"{task}_{id_}_{idx_}" not in rm_dup_task:
            rm_dup_task[f"{task}_{id_}_{idx_}"] = 1
        else:
            dump_task.append(f"{task}_{id_}_{idx_}")
            continue
        if args.skip_none_answer and ("time out" in json.dumps(item["predict_result"]) or (
                item["predict_result"][-1]["content"] is None
                and item["predict_result"][-1]["tool_calls"] is None
        ) or ("filter_by_prompt" in json.dumps(item["predict_result"]))):
            time_out_cnt += 1
            continue

        if "xlam" in args.data_file:
            try:
                if item["predict_result"] is not None and len(item["predict_result"]) == len(
                        item["answer_list"]) - 1 and "tool_calls" in item["predict_result"][0] and \
                        item["predict_result"][0]['tool_calls'] is not None and len(
                        item["predict_result"][0]['tool_calls']) != 0 and (
                        item["predict_result"][0]["tool_calls"][0]["function"]["name"] ==
                        item["answer_list"][0]["action"]["name"]
                ):
                    item["predict_label"] = "correct"
            except Exception as e:
                print(f"error: {e}")
                pass

        bad_function_pair = {}
        all_error_type = None
        if args.parameters_eval:
            item["param_predict_label"] = item["predict_label"]
            if item["param_predict_label"] == "correct":
                try:
                    flag_, bad_function_pair, all_error_type = check_every_function_arguments(item["answer_list"],
                                                                                              item["predict_result"],
                                                                                              item["tools"],
                                                                                              item_id={"id": item["id"],
                                                                                                       "idx": item["idx"]})
                except Exception as e:
                    print(f"error: {e}")
                    flag_ = True
                    error_arguments += 1
                    item["predict_label"] = "error"
                    item["param_predict_label"] = "error"
                if not flag_:
                    bad_function_pairs.append(bad_function_pair)
                    item["param_predict_label"] = "error"
                    error_arguments += 1
                    assert any([len(_["error_type"]) for _ in all_error_type]), "{} {}".format(item["idx"], item["id"])

        task_index = item["idx"]

        task_count[task_index] = task_count.get(task_index, []) + [{
            'predict_label': item["predict_label"],
            'predict_is_optimal': item["predict_is_optimal"],
        }]

        task_type[item["type"][task_index]] = task_type.get(item["type"][task_index], []) + [{
            'predict_label': item["predict_label"],
            'predict_is_optimal': item["predict_is_optimal"]
        }]

        if task_index not in task_type_count:
            task_type_count[task_index] = {}

        if item["type"][task_index] not in task_type_count[task_index]:
            task_type_count[task_index][item["type"][task_index]] = []

        if "".join(item["type"][:task_index + 1]) not in matrix_map["route_matrix"]:
            matrix_map["route_matrix"]["".join(item["type"][:task_index + 1])] = []

        matrix_map["route_matrix"]["".join(item["type"][:task_index + 1])].append(0)

        task_type_count[task_index][item["type"][task_index]].append({
            'predict_label': item["predict_label"],
            'predict_is_optimal': item["predict_is_optimal"]
        })

        if "messages_length" in item:
            messages_length = item["messages_length"]
        else:
            messages_length = -1
            for idx in range(len(item["messages"])):
                if item["messages"][idx]["content"] == item["task"]:
                    messages_length = idx

        matrix_map["num"] = weight_matrix_auto(matrix_map["num"], task_index, item["type"], item["predict_label"])

        if args.weight_eval:
            matrix_map["weight"] = weight_matrix_auto(matrix_map["weight"], task_index, item["type"],
                                                      item["predict_label"])

        if args.parameters_eval:
            matrix_map["params"] = weight_matrix_auto(matrix_map["params"], task_index, item["type"],
                                                      item["param_predict_label"])

        # messages length analysis
        if messages_length not in matrix_map["messages"]:
            matrix_map["messages"][messages_length] = [0, 0]
        matrix_map["messages"][messages_length][1] += 1

        # multi tools optical analysis
        multi_tool_task_type = analysis_answer_type(item["answer_list"])
        if item["predict_label"] != "correct" or item["param_predict_label"] != "correct":
            error_type = check_error_type(item, all_error_type)
            matrix_map["error_type"][error_type] = matrix_map["error_type"].get(error_type, 0) + 1
            item["error_type"] = error_type
        else:
            item["error_type"] = None

        if multi_tool_task_type in ["combine", "serial", "cs"]:
            matrix_map[f"{multi_tool_task_type}_tools_rate"][1] += 1

        # turn type analysis
        turn_type = str(
            item["turn_type"][task_index] == "真"
            if type(item["turn_type"][task_index]) == str
            else item["turn_type"][task_index]
        ) if "turn_type" in item and len(item["turn_type"]) != 0 else "False"

        matrix_map["turn_type_rate"][turn_type][1] += 1
        subturn_type = None

        if turn_type == "True":
            if len(item["turn_subtypes"]) != len(item["turn_type"]):
                assert len(item["turn_subtypes"]) == len([_ for _ in item["turn_type"] if _])
                turn_subtype = []
                turn_subtype_idx = 0
                for tty in item["turn_type"]:
                    if tty:
                        turn_subtype.append(item["turn_subtypes"][turn_subtype_idx])
                        turn_subtype_idx += 1
                    else:
                        turn_subtype.append(None)
                item["turn_subtypes"] = turn_subtype

            subturn_type = (
                item["turn_subtypes"][task_index]
                if 'turn_subtypes' in item and len(item["turn_subtypes"]) != 0
                else "长期记忆"
            )
            matrix_map["turn_subtype_rate"][subturn_type][1] += 1

        item["turn_type"] = turn_type
        item["turn_subtype"] = subturn_type
        if multi_tool_task_type in ["combine", "cs"]:
            matrix_map["best_optical_rate"]["sum"][1] += 1
            matrix_map["best_optical_rate"][multi_tool_task_type][1] += 1

        if (
                args.parameters_eval and item["param_predict_label"] == "correct"
        ) or (
                not args.parameters_eval and item["predict_label"] == "correct"
        ):
            matrix_map["messages"][messages_length][0] += 1
            if multi_tool_task_type in ["combine", "serial", "cs"]:
                matrix_map[f"{multi_tool_task_type}_tools_rate"][0] += 1

            if multi_tool_task_type in ["combine", "cs"] and item["predict_is_optimal"] == "True":
                matrix_map["best_optical_rate"]["sum"][0] += 1
                matrix_map["best_optical_rate"][multi_tool_task_type][0] += 1

            matrix_map["route_matrix"]["".join(item["type"][:task_index + 1])][-1] = 1
            matrix_map["turn_type_rate"][turn_type][0] += 1

            if subturn_type is not None:
                matrix_map["turn_subtype_rate"][subturn_type][0] += 1

        complete_rate = None
        if multi_tool_task_type in ["serial", "cs"]:
            matrix_map["serial_complete_rate"]["sum"][1] += len(item["answer_list"])
            matrix_map["serial_complete_rate"][multi_tool_task_type][1] += len(item["answer_list"])
            complete_rate = calculate_predict_result_steps(
                item["param_predict_label"],
                item["predict_result"],
                item["answer_list"],
                bad_function_pair
            )
            matrix_map["serial_complete_rate"]["sum"][0] += complete_rate
            matrix_map["serial_complete_rate"][multi_tool_task_type][0] += complete_rate

        item["complete_rate"] = complete_rate
        item["single_type"] = item["type"][task_index]

    final_metric_dict = {}
    final_metric_dict["case_num"] = len(rm_dup_task)
    print("SUM:", len(rm_dup_task))
    print("time_out_cnt:", time_out_cnt)
    print("dump_task:", dump_task)
    print("error_arguments:", error_arguments)
    print("origin")
    print("=" * 60)

    final_metric_dict["num"] = draw_matrix_by_type_and_index(matrix_map["num"], args.answer_depth)
    if args.weight_eval:
        print("Weight")
        print("=" * 60)
        final_metric_dict["weight"] = draw_matrix_by_type_and_index(matrix_map["weight"], args.answer_depth)

    if args.parameters_eval:
        print("Params")
        print("=" * 60)
        final_metric_dict["params"] = draw_matrix_by_type_and_index(matrix_map["params"], args.answer_depth)

    print("Multi tools analysis:")
    print("=" * 60)
    for multi_tool_task_type in ["combine", "serial", "cs"]:
        print("{}_tools_rate: {}/{}={}".format(
            multi_tool_task_type,
            matrix_map[f"{multi_tool_task_type}_tools_rate"][0],
            matrix_map[f"{multi_tool_task_type}_tools_rate"][1],
            matrix_map[f"{multi_tool_task_type}_tools_rate"][0] / (
                        matrix_map[f"{multi_tool_task_type}_tools_rate"][1] + 1e-5)
        ))
        final_metric_dict[f"{multi_tool_task_type}_tools_rate"] = matrix_map[f"{multi_tool_task_type}_tools_rate"][
                                                                      0] / (matrix_map[
                                                                                f"{multi_tool_task_type}_tools_rate"][
                                                                                1] + 1e-5)
    final_metric_dict["serial_complete_rate"] = {}
    for k in ["sum", "serial", "cs"]:
        print("serial_complete_rate {}: {}/{}={}".format(
            k,
            matrix_map["serial_complete_rate"][k][0],
            matrix_map["serial_complete_rate"][k][1],
            matrix_map["serial_complete_rate"][k][0] / (1e-5 + matrix_map["serial_complete_rate"][k][1])
        ))
        final_metric_dict["serial_complete_rate"][k] = matrix_map["serial_complete_rate"][k][0] / (
                    1e-5 + matrix_map["serial_complete_rate"][k][1])

    final_metric_dict["best_optical_rate"] = {}
    for k in ["sum", "combine", "cs"]:
        print("best_optical_rate {}: {}/{}={}".format(
            k,
            matrix_map["best_optical_rate"][k][0],
            matrix_map["best_optical_rate"][k][1],
            matrix_map["best_optical_rate"][k][0] / (1e-5 + matrix_map["best_optical_rate"][k][1])
        ))
        final_metric_dict["best_optical_rate"][k] = matrix_map["best_optical_rate"][k][0] / (
                    1e-5 + matrix_map["best_optical_rate"][k][1])

    matrix_calculate_variance(matrix_map["route_matrix"])
    print("\nerror type analysis")
    print("=" * 60)
    value = []
    value_action = []
    value_param = []
    sum_value = sum([_ for _ in matrix_map["error_type"].values()])
    sum_value_action = sum([matrix_map["error_type"].get(k, 0) for k in [
        "reject tool calling",
        "function name hallucination",
        "missing information in tool calling",
        "redundant tool calling",
        "call function without reject",
        "call the wrong function",
        "early stop",
    ]])
    sum_value_param = sum_value - sum_value_action
    final_metric_dict["error_type_analysis"] = {}

    for idx, k in enumerate([
        "reject tool calling",
        "function name hallucination",
        "missing information in tool calling",
        "redundant tool calling",
        "call function without reject",
        "call the wrong function",
        "early stop",
        "miss required params",
        "param hallucination",
        "type error",
        "param value hallucination",
        "param value error",
    ]):
        val = "{:.06}".format(matrix_map["error_type"].get(k, 0) / (sum_value + 1e-5))
        if idx <= 5:
            val_action = "{:.06}".format(matrix_map["error_type"].get(k, 0) / (sum_value_action + 1e-5))
            value_action.append(val_action)
        else:
            val_param = "{:.06}".format(matrix_map["error_type"].get(k, 0) / (sum_value_param + 1e-5))
            value_param.append(val_param)
        print(k, "\t", val)
        value.append(val)
        final_metric_dict["error_type_analysis"][k] = float(val)

    print("-" * 10)
    print(" ".join(value))
    print("action:", " ".join(value_action))
    print("param:", " ".join(value_param))

    triangle_data = matrix_calculate_triangle_data(matrix_map["route_matrix"])
    with open("./triangle_data/{}".format(args.data_file.split("/")[-1]), "w") as f:
        f.write(json.dumps(triangle_data))
    if args.show_triangle:
        print(json.dumps(triangle_data))

    print("\nturn type analysis:")
    final_metric_dict["turn_type_rate"] = {}
    final_metric_dict["turn_subtype_rate"] = {}
    for turn_type in matrix_map["turn_type_rate"]:
        val = matrix_map["turn_type_rate"][turn_type][0] / (matrix_map["turn_type_rate"][turn_type][1] + 1e-5)
        print(turn_type, matrix_map["turn_type_rate"][turn_type][0], matrix_map["turn_type_rate"][turn_type][1], val)
        final_metric_dict["turn_type_rate"][turn_type] = val

    print("\nsubturn type analysis:")
    for turn_type in matrix_map["turn_subtype_rate"]:
        val = matrix_map["turn_subtype_rate"][turn_type][0] / (matrix_map["turn_subtype_rate"][turn_type][1] + 1e-5)
        print(turn_type, matrix_map["turn_subtype_rate"][turn_type][0], matrix_map["turn_subtype_rate"][turn_type][1],
              val)
        final_metric_dict["turn_subtype_rate"][turn_type] = val

    # sorted by message length and print rate
    print("\nmessages_by length:")
    sorted_messages_length = sorted(matrix_map["messages"].items(), key=lambda x: x[0])
    for length, cnt in sorted_messages_length:
        if cnt[1] < 10:
            continue
        print(length, cnt[1], cnt[0] / cnt[1])
    print("samples (<10) have been removed.")
    param_error_type_analysis(bad_function_pairs)
    if args.badpair_path is not None:
        write_json_to_file(bad_function_pairs, args.badpair_path, print_f=False)

    if args.output_csv_flag:
        keys = ['id', 'idx', 'type', 'single_type', 'turn_type', 'turn_subtype', 'messages_length', 'tools', 'messages',
                'answer_list', 'answer_result', 'predict_result', 'complete_rate', 'error_type', 'param_predict_label',
                'predict_is_optimal', 'predict_label']
        list_of_data = [keys]
        for item in data2:
            item_ = []
            if "turn_subtype" not in item:
                continue

            for k in keys:
                item_.append(item[k] if type(item[k]) == str else json.dumps(
                    item[k], ensure_ascii=False, indent=2
                ))
            list_of_data.append(item_)

        write_list_of_list_to_csv(list_of_data, args.output_csv_path)

    return final_metric_dict


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
        new_messages += messages[:task_id + 1]
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
                    m['content'] = m["content"][colon_idx + 1:]
            else:
                colon_idx = m["content"].find("：")
                if (
                        colon_idx != -1 and
                        m["content"][:colon_idx] in [
                    "用户", "AI Agent助手", "AI Agent"
                ]
                ):
                    m['content'] = m["content"][colon_idx + 1:]
    except Exception as e:
        print(f"error: {e}")
        # ipdb.set_trace()

    return new_messages


def get_value_from_dict_to_list(value_dict):
    if type(value_dict) == str:
        return value_dict

    only_keep_keys = [
        "case_num",
        "||",
        "params",
        "||",
        "combine_tools_rate",
        "serial_tools_rate",
        "cs_tools_rate",
        "||",
        "serial_complete_rate,serial",
        "serial_complete_rate,cs",
        "serial_complete_rate,sum",
        "||",
        "best_optical_rate,combine",
        "best_optical_rate,cs",
        "best_optical_rate,sum",
        "||",
        "error_type_analysis,reject tool calling",
        "error_type_analysis,function name hallucination",
        "error_type_analysis,missing information in tool calling",
        "error_type_analysis,call function without reject",
        "error_type_analysis,redundant tool calling",
        "error_type_analysis,call the wrong function",
        "error_type_analysis,early stop",
        "error_type_analysis,miss required params",
        "error_type_analysis,param hallucination",
        "error_type_analysis,type error",
        "error_type_analysis,param value hallucination",
        "error_type_analysis,param value error",
        "turn_type_rate,False",
        "turn_type_rate,True",
        "turn_subtype_rate,指代理解",
        "turn_subtype_rate,省略成分",
        "turn_subtype_rate,长期记忆",
    ]
    value_list = []

    def add_val(value_list, val):
        if type(val) == list:
            value_list += val
        elif type(val) == dict:
            value_list += [f"{k}:{v}" for k, v in val.items()]
        else:
            value_list.append(val)

    for key in only_keep_keys:
        if key == "||":
            value_list.append("||")
        if key in value_dict:
            add_val(value_list, value_dict[key])
        else:
            if "," in key:
                def get_key_value_by_route(dict, key_list):
                    if len(key_list) == 0:
                        return dict
                    if key_list[0] not in dict:
                        return None
                    else:
                        return get_key_value_by_route(dict[key_list[0]], key_list[1:])

                add_val(value_list, get_key_value_by_route(value_dict, key.split(",")))

    return value_list


def analysis_all_file_in_path(args):
    if "," in args.data_file:
        data_file_path = ""
        all_files = args.data_file.split(",")
    else:
        data_file_path = args.data_file
        all_files = os.listdir(data_file_path)

    metric_map = {}
    for file_ in all_files:
        try:
            if not file_.endswith(".jsonl") and not file_.endswith(".jsonl.merge"):
                continue
            args.data_file = os.path.join(data_file_path, file_)
            metric_map[file_] = main2(args)
        except Exception as e:
            print(f"error: ", file_, "\n", e)
            metric_map[file_] = e

    metric_map_reform = {}
    for file_, value_dict in metric_map.items():
        file_ = file_[file_.find("_") + len("_1a8e65_"):file_.rfind("_en") if "_en_" in file_ else file_.rfind("_zh")]
        metric_map_reform[file_] = get_value_from_dict_to_list(value_dict)

    already_keys = [
        "gpt4",
        "gemini",
        "claude",
        "mistral",
        "doubao",
        "deepseekv3",
        "toolace",
        "xlam",
        "gorilla",
        "watt70b",
        "watt8b",
        "hammer7b",
        "hammer3b",
        "hammer1.5b",
        "hammer0.5b",
        "fcm3.1",
        "llama70b",
        "llama8b",
        "llama3b",
        "llama1b",
        "qwen72b",
        "qwen32b",
        "qwen14b",
        "qwen7b",
        "qwen3b",
        "qwen1.5b",
        "qwen0.5b",
        "chatglm",
    ]
    for model_name in already_keys:
        if model_name not in metric_map_reform:
            print(model_name, "Miss")
        else:
            print(model_name, " ".join([str(_) for _ in metric_map_reform[model_name]]))

    sorted_model_key = sorted(metric_map_reform.keys(), reverse=True)
    print("KEYS:", " ".join(sorted_model_key))
    for key in sorted_model_key:
        if key not in already_keys:
            print(key, " ".join([str(_) for _ in metric_map_reform[key]]))


if __name__ == "__main__":
    args = parse_argument()
    if os.path.isfile(args.data_file) and "," not in args.data_file:
        res = main2(args)
        print(" ".join([str(_) for _ in get_value_from_dict_to_list(res)]))
    else:
        analysis_all_file_in_path(args)
