import traceback
import ast

from mmtb.bench_test.utils import get_keywords


class AstVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function = []

    def visit_Call(self, node):
        # self.function_name, self.args = parse_string_to_function(node)
        function = {}
        if isinstance(node.func, ast.Name):
            function["name"] = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function["name"] = node.func.attr

        function["arguments"] = {}
        for keyword in node.keywords:
            function["arguments"][keyword.arg] = get_keywords(keyword.value)
        self.function.append(function)

    def clear(self):
        self.function = []


def remove_messages(messages, is_english=False):
    new_messages = []
    try:
        role = "user"
        for m in messages:
            assert (
                m["role"] == "assistant"
                and role == "assistant"
            ) or (
                m["role"] in ["user", "tool"]
                and role in ["user", "tool"]
            )
            role = "assistant" if role in ["user", "tool"] else "user"
            if is_english:
                colon_idx = m["content"].find(":")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx].lower() in [
                        "ai", "ai agent", "user", "ai agent assistant", "planner", "observation", "tool"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
            else:
                colon_idx = m["content"].find("：")
                if (
                    colon_idx != -1 and
                    m["content"][:colon_idx] in [
                        "用户", "AI Agent助手", "AI Agent", "Planner", "Observation", "Tool"
                    ]
                ):
                    m['content'] = m["content"][colon_idx+1:]
            new_messages.append(m)
    except Exception as e:
        print(f"error: {e}")
        traceback.print_exc()
    return new_messages
