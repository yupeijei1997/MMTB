import json
import copy
import concurrent.futures
import time

from handle.gpt_handle import GPTAZUREMultiTurnMessages
from itertools import combinations


class ToolCallNode:
    def __init__(self, node_action, node_observation, node_user_input, node_dependency_list):
        self.action = node_action
        self.observation = node_observation
        self.user_input = node_user_input
        self.dependency_list = node_dependency_list


class ToolCallGraph:
    def __init__(self, answer_list):
        self.answer_list = answer_list
        self.node_list = []
        self.all_path_list = []
        self.optimal_path_list = []
        self.suboptimal_path_list = []
        self.updating_all_path_list = []
        self.min_length = 99999

    def add_node(self, tool_call_node):
        self.node_list.append(tool_call_node)

    def add_node_list(self):
        for i, answer in enumerate(self.answer_list):
            action = answer["action"]
            observation = answer["observation"]
            user_input = answer.get("user_input", None)
            dependency_list = answer["dependency_list"]
            tool_call_node = ToolCallNode(action, observation, user_input, dependency_list)
            self.add_node(tool_call_node)

    def generate_all_path(self):
        print("generate path")
        start_time = time.time()
        self.init_graph_and_in_degree()
        print("init graph and in degree: ", time.time() - start_time)
        self.dfs(self.graph, self.in_degree, [False] * len(self.node_list), [])
        print("generate path: ", time.time() - start_time)
        self.split_path()
        print("split path: ", time.time() - start_time)
        self.init_step_to_answer()
        print("init step to answer: ", time.time() - start_time)

    def init_graph_and_in_degree(self):
        """
        初始化图、入度表
        """
        self.graph = {i: [] for i in range(len(self.node_list))}
        self.in_degree = {i: 0 for i in range(len(self.node_list))}
        # 构建图和入度表
        for idx, node in enumerate(self.node_list):
            dependency_list = node.dependency_list
            for dependency in dependency_list:
                self.graph[dependency].append(idx)
                self.in_degree[idx] += 1

    def dfs(self, graph, in_degree, visited, path):
        """
        执行拓扑排序并生成路径
        """
        node_nums = 0
        for p in path:
            node_nums += len(p)
        if node_nums == len(self.answer_list):
            self.all_path_list.append(copy.deepcopy(path))
        else:
            # 初始化列表，包含所有入度为0的节点
            current_zero_in_degree_node_list = [node for node in in_degree if
                                                in_degree[node] == 0 and not visited[node]]

            # 初始化结果列表
            continuous_same_function_name_node_list = []
            # 初始化当前连续字母列表
            current_sequence = []
            last_function_name = ""
            for i, node_idx in enumerate(current_zero_in_degree_node_list):
                node = self.node_list[node_idx]
                function_name = node.action["name"]
                if i == 0 or function_name == last_function_name:
                    # 如果是第一个元素或者当前元素与前一个元素相同
                    current_sequence.append(node_idx)
                else:
                    # 如果当前元素与前一个元素不同，那么结束当前连续字母列表
                    # 并将其添加到结果列表中
                    continuous_same_function_name_node_list.append(current_sequence)
                    current_sequence = [node_idx]
                last_function_name = function_name

            if current_sequence:
                continuous_same_function_name_node_list.append(current_sequence)

            need_filter_combinations = []
            not_need_filter_combinations = []
            for node_list in continuous_same_function_name_node_list:
                if len(node_list) > 1:
                    for r in range(1, len(node_list)):
                        need_filter_combinations.extend(combinations(node_list, r))
                    not_need_filter_combinations.append(tuple(node_list))

            # 生成所有可能的组合
            all_combinations = []
            for r in range(1, len(current_zero_in_degree_node_list) + 1):
                all_combinations.extend(combinations(current_zero_in_degree_node_list, r))

            all_combinations_deduplication = []
            all_combinations_by_function_name_set = set()
            for comb in all_combinations:
                function_name_list = []
                for c in comb:
                    node = self.node_list[c]
                    function_name = node.action["name"]
                    function_name_list.append(function_name)
                function_name_str = "|".join(function_name_list)
                if function_name_str not in all_combinations_by_function_name_set:
                    all_combinations_by_function_name_set.add(function_name_str)
                    all_combinations_deduplication.append(comb)

            all_combinations_final = []
            for comb in all_combinations_deduplication:
                filter_flag = False
                if len(comb) > 1:
                    if comb in need_filter_combinations:
                        filter_flag = True

                    sub_comb_list = [comb[:i + 1] for i in range(len(comb))]
                    for sub_comb in sub_comb_list:
                        if sub_comb in need_filter_combinations:
                            filter_flag = True

                    for sub_comb in sub_comb_list:
                        if sub_comb in not_need_filter_combinations:
                            filter_flag = False

                    if filter_flag:
                        continue

                all_combinations_final.append(comb)

            all_combinations = all_combinations_final

            for comb in all_combinations:
                extend_flag = False
                if len(comb) == 1:
                    for not_comb in not_need_filter_combinations:
                        if comb[0] == not_comb[0]:
                            comb = copy.deepcopy(not_comb)
                            extend_flag = True
                            break

                visited_copy = copy.deepcopy(visited)
                comb = list(comb)
                for c in comb:
                    visited_copy[c] = True
                in_degree_copy = copy.deepcopy(in_degree)
                path_copy = copy.deepcopy(path)
                for node in comb:
                    # 更新入度表
                    for neighbor in graph[node]:
                        in_degree_copy[neighbor] -= 1

                if extend_flag:
                    for c in comb:
                        path_copy.append([c])
                else:
                    path_copy.append(comb)
                self.dfs(graph, in_degree_copy, visited_copy, path_copy)

    def split_path(self):
        for path in self.all_path_list:
            if len(path) < self.min_length:
                self.min_length = len(path)

        for path in self.all_path_list:
            if len(path) == self.min_length:
                self.optimal_path_list.append(path)
            else:
                self.suboptimal_path_list.append(path)

        self.updating_all_path_list = copy.deepcopy(self.all_path_list)

    def init_step_to_answer(self):
        self.step_to_idx_list = {i: [] for i in range(len(self.node_list))}
        self.step_to_function_name_list = {i: [] for i in range(len(self.node_list))}
        self.step_to_function_arguments_list = {i: [] for i in range(len(self.node_list))}
        self.step_to_function_observation_list = {i: [] for i in range(len(self.node_list))}
        self.step_to_user_input_list = {i: [] for i in range(len(self.node_list))}
        for path in self.updating_all_path_list:
            for step, idx_list in enumerate(path):
                self.step_to_idx_list[step].append(idx_list)
                function_name_list = []
                function_arguments_list = []
                function_observation_list = []
                user_input_list = []
                for i, idx in enumerate(idx_list):
                    node = self.node_list[idx]
                    node_action = node.action
                    function_name = node_action["name"]
                    function_arguments = node_action["arguments"]
                    function_observation = node.observation
                    user_input = node.user_input

                    function_name_list.append(function_name)
                    function_arguments_list.append(function_arguments)
                    function_observation_list.append(function_observation)
                    user_input_list.append(user_input)

                idx_function_name_list = list(enumerate(function_name_list))
                sorted_idx_function_name_list = sorted(idx_function_name_list, key=lambda x: x[1])
                sorted_indices = [idx for idx, function_name in sorted_idx_function_name_list]
                sorted_function_name_list = [function_name_list[i] for i in sorted_indices]
                sorted_function_arguments_list = [function_arguments_list[i] for i in sorted_indices]
                sorted_function_observation_list = [function_observation_list[i] for i in sorted_indices]
                sorted_user_input_list = [user_input_list[i] for i in sorted_indices]

                self.step_to_function_name_list[step].append(sorted_function_name_list)
                self.step_to_function_arguments_list[step].append(sorted_function_arguments_list)
                self.step_to_function_observation_list[step].append(sorted_function_observation_list)
                self.step_to_user_input_list[step].append(sorted_user_input_list)

    def update_updating_all_path_list(self, step, idx_to_list):
        new_updating_all_path_list = []
        for path in self.updating_all_path_list:
            current_step_path = path[step]
            if idx_to_list != current_step_path:
                continue
            else:
                new_updating_all_path_list.append(path)
        self.updating_all_path_list = new_updating_all_path_list


def run_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function '{func.__name__}' exceededtimeout of {timeout} seconds")


def eval_by_tool_call_graph(request_func, messages, tools, answer_list, continue_tool_message, env_info=None, retry_num=1):
    # print("generate", len(answer_list))
    tool_call_graph = ToolCallGraph(answer_list)
    try:
        run_with_timeout(tool_call_graph.add_node_list, 60)
        run_with_timeout(tool_call_graph.generate_all_path, 60)
    except:
        return "graph generate timeout.", None, None, None

    step = 0
    label = "error"
    predict_result = []
    answer_result = []
    while True:
        # print(f"step: {step}\n")
        # print(f"输入：")
        # for message in messages:
        #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n")
        test_time = 0
        content = None
        tool_calls = None
        while test_time < retry_num:
            try:
                test_time += 1
                messages_tmp = copy.copy(messages)
                tools_tmp = copy.copy(tools)
                env_info_ = copy.copy(env_info)
                content, tool_calls = request_func(messages_tmp, tools_tmp, env_info_)
                time.sleep(3)
                if content is not None or tool_calls is not None:
                    break
            except Exception as e:
                print(e)
                test_time += 1
                time.sleep(3)
                continue
        # print(f"输出：")
        # print(f"content: \n{content}\n")
        # print(f"tool_calls: \n{json.dumps(tool_calls, ensure_ascii=False, indent=4)}\n")
        predict_result.append(
            {"step": step, "content": content, "tool_calls": tool_calls}
        )
        if not tool_calls:
            if not content:
                label = "error"
                break
            else:
                current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                for i, (answer_function_name_list, answer_function_arguments_list) in enumerate(zip(current_step_function_name_list, current_step_function_arguments_list)):
                    if "ask_user_for_required_parameters" in answer_function_name_list:
                        messages.append(
                            {"role": "assistant", "content": content}
                        )

                        function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                        assert len(function_observation_list) == 1
                        function_observation = function_observation_list[0]

                        user_input_list = tool_call_graph.step_to_user_input_list[step][i]
                        assert len(user_input_list) == 1
                        user_input = user_input_list[0]
                        messages.append(
                            {"role": "user", "content": user_input}
                        )

                        answer_function_list = {"action": [], "observation": function_observation, "user_input": user_input}
                        for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                            answer_function_list["action"].append({"name": answer_function_name, "arguments": answer_function_arguments})
                        answer_result.append({"step": step, "answer_function_list": answer_function_list})

                        break
                    elif "prepare_to_answer" in answer_function_name_list:
                        function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                        assert len(function_observation_list) == 1
                        function_observation = function_observation_list[0]

                        messages.append(
                            {"role": "assistant", "content": content}
                        )
                        label = "correct"

                        answer_function_list = {"action": [], "observation": function_observation}
                        for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                            answer_function_list["action"].append({"name": answer_function_name, "arguments": answer_function_arguments})
                        answer_result.append({"step": step, "answer_function_list": answer_function_list})

                        break
                else:
                    label = "error"
                    break

        else:
            tool_calls_len = len(tool_calls)
            predict_function_name_list = []
            predict_function_arguments_list = []
            predict_function_id_list = []
            try:
                for tool_call in tool_calls:
                    function_id = tool_call["id"]
                    function = tool_call["function"]
                    function_name = function["name"]
                    function_arguments = function["arguments"]
                    predict_function_id_list.append(function_id)
                    predict_function_name_list.append(function_name)
                    predict_function_arguments_list.append(function_arguments)
            except:
                print(f"ERROR: {tool_calls}")
                label = "error"
                break

            idx_predict_function_name_list = list(enumerate(predict_function_name_list))
            sorted_idx_predict_function_name_list = sorted(idx_predict_function_name_list, key=lambda x: x[1])
            sorted_indices = [idx for idx, predict_function_name in sorted_idx_predict_function_name_list]
            sorted_predict_function_name_list = [predict_function_name_list[i] for i in sorted_indices]
            sorted_predict_function_arguments_list = [predict_function_arguments_list[i] for i in sorted_indices]
            predict_function_name_list = sorted_predict_function_name_list
            predict_function_arguments_list = sorted_predict_function_arguments_list

            idx_list = tool_call_graph.step_to_idx_list.get(step, None)
            if not idx_list:
                label = "error"
                break
            else:
                current_step_function_name_list = tool_call_graph.step_to_function_name_list[step]
                current_step_function_arguments_list = tool_call_graph.step_to_function_arguments_list[step]
                for i, (answer_function_name_list, answer_function_arguments_list) in enumerate(zip(current_step_function_name_list, current_step_function_arguments_list)):
                    if predict_function_name_list != answer_function_name_list:
                        continue
                    else:
                        messages.append(
                            {"role": "assistant", "content": content, "tool_calls": tool_calls}
                        )

                        function_observation_list = tool_call_graph.step_to_function_observation_list[step][i]
                        if continue_tool_message:
                            # 支持连续的多个tool message
                            for j, function_observation in enumerate(function_observation_list):
                                if not isinstance(function_observation, str):
                                    function_observation = json.dumps(function_observation, ensure_ascii=False)
                                function_id = predict_function_id_list[j]
                                messages.append(
                                    {"role": "tool", "content": function_observation, "tool_call_id": function_id}
                                )
                        else:
                            # 不支持连续的多个tool message
                            function_observation_list = json.dumps(function_observation_list, ensure_ascii=False)
                            function_id = predict_function_id_list[0]
                            messages.append(
                                {"role": "tool", "content": function_observation_list, "tool_call_id": function_id}
                            )

                        # 剪枝
                        idx_list = tool_call_graph.step_to_idx_list[step][i]
                        tool_call_graph.update_updating_all_path_list(step, idx_list)
                        tool_call_graph.init_step_to_answer()

                        answer_function_list = {"action": []}
                        for answer_function_name, answer_function_arguments in zip(answer_function_name_list, answer_function_arguments_list):
                            answer_function_list["action"].append({"name": answer_function_name, "arguments": answer_function_arguments})
                        answer_result.append({"step": step, "answer_function_list": answer_function_list})

                        break
                else:
                    label = "error"
                    break

        if label == "correct":
            break

        step += 1

    print(f"infer end\n")
    # for message in messages:
    #     print(json.dumps(message, ensure_ascii=False, indent=4) + "\n")
    print(f"label: {label}\n")
    if label == "correct":
        if step == (tool_call_graph.min_length - 1):
            is_optimal = True
        else:
            is_optimal = False
    else:
        is_optimal = False
    return label, is_optimal, predict_result, answer_result


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_text_page",
                "description": "添加PPT中的文本页",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "页面的标题"
                        },
                        "bullet_items": {
                            "type": "string",
                            "description": "bullet项应为字符串，对于多个bullet项，请使用[SPAN]进行分隔"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_file",
                "description": "当所有步骤完成后，您必须使用save_file()来保存PPT",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "": {

                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "serper_search",
                "description": "一个谷歌搜索API。当你想要得到关于特定问题的答案时，你可以使用它",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询的内容"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_movie_reviews",
                "description": "获取电影的用户评论",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "movie_name": {
                            "type": "string",
                            "description": "电影名"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "使用特定主题创建一个pptx文件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "使用的主题"
                        },
                        "abs_location": {
                            "type": "string",
                            "description": "ppt文件的绝对保存位置"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_rated_movies",
                "description": "按评分顺序获取电影列表",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "": {

                        }
                    }
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "给我做一个ppt文件movie_reviews.ppt，地址在/documents/，内容包含排名第一的电影的姓名、评分和评论内容"}
    ]

    answer_list = \
        [
            {
                "idx": 0,
                "action": {
                    "name": "create_file",
                    "arguments": {
                        "abs_location": "/documents/movie_reviews.ppt",
                        "theme": "电影评论"
                    }
                },
                "observation": {
                    "status": "Success",
                    "message": "The PPT file movie_reviews.ppt with Classic theme has been successfully created."
                },
                "dependency_list": []
            },
            {
                "idx": 1,
                "action": {
                    "name": "get_top_rated_movies",
                    "arguments": {}
                },
                "observation": [
                    {
                        "movie_title": "The Shawshank Redemption",
                        "rating": 9.3,
                        "release_year": 1994
                    },
                    {
                        "movie_title": "The Godfather",
                        "rating": 9.2,
                        "release_year": 1972
                    },
                    {
                        "movie_title": "The Godfather: Part II",
                        "rating": 9,
                        "release_year": 1974
                    },
                    {
                        "movie_title": "Pulp Fiction",
                        "rating": 8.9,
                        "release_year": 1994
                    },
                    {
                        "movie_title": "The Good, the Bad and the Ugly",
                        "rating": 8.8,
                        "release_year": 1966
                    }
                ],
                "dependency_list": []
            },
            {
                "idx": 2,
                "action": {
                    "name": "get_movie_reviews",
                    "arguments": {
                        "movie_name": "The Shawshank Redemption"
                    }
                },
                "observation": {
                    "movie_name": "The Shawshank Redemption",
                    "reviews": [
                        {
                            "user": "John Doe",
                            "rating": "5/5",
                            "comment": "Absolutely brilliant! The characters, the plot and the entire setup is amazing."
                        },
                        {
                            "user": "Jane Doe",
                            "rating": "4/5",
                            "comment": "A breathtaking story of hope and friendship. A must watch."
                        },
                        {
                            "user": "Sam Smith",
                            "rating": "5/5",
                            "comment": "Arguably one of the best movies of all time! Highly recommend."
                        },
                        {
                            "user": "Alex Johnson",
                            "rating": "5/5",
                            "comment": "The acting by Tim Robbins and Morgan Freeman is outstanding. The story is intriguing and captivating. Amazing movie."
                        }
                    ]
                },
                "dependency_list": [1]
            },
            {
                "idx": 3,
                "action": {
                    "name": "add_text_page",
                    "arguments": {
                        "title": "The Shawshank Redemption",
                        "bullet_items": "Rating: 9.3\nRelease Year: 1994\n\nReviews:\n- John Doe (5/5): Absolutely brilliant! The characters, the plot and the entire setup is amazing.\n- Jane Doe (4/5): A breathtaking story of hope and friendship. A must watch.\n- Sam Smith (5/5): Arguably one of the best movies of all time! Highly recommend.\n- Alex Johnson (5/5): The acting by Tim Robbins and Morgan Freeman is outstanding. The story is intriguing and captivating. Amazing movie."
                    }
                },
                "observation": {
                    "status": "success",
                    "message": "Text page successfully added to the PPT",
                    "page_id": "002"
                },
                "dependency_list": [0, 2]
            },
            {
                "idx": 4,
                "action": {
                    "name": "save_file",
                    "arguments": {}
                },
                "observation": "PowerPoint presentation has been successfully saved.",
                "dependency_list": [3]
            },
            {
                "idx": 5,
                "action": {
                    "name": "prepare_to_answer",
                    "arguments": {
                        "answer_type": "tool"
                    }
                },
                "observation": "我已经成功创建了一个名为 **movie_reviews.ppt** 的PPT文件，内容包括排名第一的电影《肖申克的救赎》（The Shawshank Redemption）的评分、发行年份以及用户评论。PPT文件已保存并准备好使用。",
                "dependency_list": [4]
            }
        ]
    handle = GPTAZUREMultiTurnMessages("")
    eval_by_tool_call_graph(handle.request_funcall, messages, tools, answer_list, ToolCallGraph)
