from utils.agent_utils import parse_answer


def remove_prepare_ask_tools(tools):
    tools_new = []
    for tool in tools:
        tool_name = tool["function"]["name"]
        if tool_name in ["prepare_to_answer", "ask_user_for_required_parameters"]:
            continue

        tools_new.append(tool)
    return tools_new


def transform_train_data(messages, tools, env_info):
    train_data_example_origin = {"tools": tools, "env_info": env_info, "messages": messages}
    FAILED = False
    messages_new = []
    for message in messages:
        content = message["content"]
        if content.startswith("切换角色为") or content.startswith("Switch"):
            continue
        elif not content.startswith("Checker"):
            messages_new.append(message)
        else:
            content_obj = parse_answer(content)
            correct = content_obj["correct"]
            if correct == "no":
                messages_new = messages_new[:-1]

    train_data_example = {"tools": tools, "env_info": env_info, "messages": messages_new}
    return FAILED, train_data_example, train_data_example_origin
