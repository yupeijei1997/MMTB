ask_user_for_help_tool = {
    "type": "function",
    "function": {
        "name": "ask_user_for_required_parameters",
        "description": "如果你认为用户任务缺失了要调用的工具中的部分必填(required)参数，需要寻求用户帮助，则调用此函数",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "解决用户任务需要调用的工具名"
                },
                "missing_required_parameters": {
                    "type": "array",
                    "description": "用户任务中缺失的工具必填参数",
                    "items": {
                        "type": "string",
                        "description": "用户任务中缺失的必填参数"
                    }
                }
            },
            "required": ["tool_name", "missing_required_parameters"]
        }
    }
}

prepare_to_answer_tool = {
    "type": "function",
    "function": {
        "name": "prepare_to_answer",
        "description": "根据上下文信息，如果你认为已经可以完成用户任务了，则调用此函数",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_type": {
                    "type": "string",
                    "description": "回答的类型，如果是根据工具调用结果对用户任务进行总结回答，则填写为tool；如果是用户任务不需要调用任何工具，可以直接回答，则填写chat",
                    "enum": ["tool", "chat"]
                }
            },
            "required": ["answer_type"]
        }
    }
}