import os
import uuid
import datetime
import argparse
import ast
import json
import traceback



def get_random_file_name(file_type, keys=None, need_time=False, need_uuid=True):
    file_name = []
    if need_time:
        today = str(datetime.datetime.now()).replace(" ", "-").split(".")[0]
        file_name.append(today)
    if need_uuid:
        file_name.append(str(uuid.uuid4())[:6])
    if keys is not None:
        file_name.append(str(keys))
    return "_".join(file_name) + f".{file_type}"


def get_random_pathname(path_, file_type, keys=None, need_time=False, need_uuid=True):
    file_name = get_random_file_name(file_type, keys, need_time, need_uuid)
    pathname = os.path.join(path_, file_name)
    return pathname


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
    

def get_keywords(value):
    if isinstance(value, ast.Str):
        value = value.s
    elif isinstance(value, ast.Num):
        value = value.n
    elif isinstance(value, ast.UnaryOp):
        if isinstance(value.op, ast.USub):
            operand = get_keywords(value.operand)
            value = -operand
    elif isinstance(value, ast.BinOp):
        left = get_keywords(value.left)
        right = get_keywords(value.right)
        if isinstance(value.op, ast.Add):
            value = left + right
        elif isinstance(value.op, ast.Sub):
            value = left - right
        elif isinstance(value.op, ast.Mult):
            value = left * right
        elif isinstance(value.op, ast.Div):
            value = left / right
    elif isinstance(value, ast.Subscript):
        value = value.slice.value
        if isinstance(value.slice, ast.Index):
            value = value.slice.value
        elif isinstance(value.slice, ast.Slice):
            value = value.slice.value
        elif isinstance(value.slice, ast.Ellipsis):
            value = "..."
    elif isinstance(value, ast.NameConstant):
        value = value.value
    elif isinstance(value, ast.Name):
        if value.id.lower() == "true":
            value = True
        elif value.id.lower() == "false":
            value = False
        else:
            value = value.id
    elif isinstance(value, ast.List):
        value = [get_keywords(elt) for elt in value.elts]
    elif isinstance(value, ast.Tuple):
        value = tuple([get_keywords(elt) for elt in value.elts])
    elif isinstance(value, ast.Dict):
        value = {
            get_keywords(key): get_keywords(val)
            for key, val in zip(value.keys, value.values)
        }
    else:
        raise Exception("Unsupported type: {}".format(type(value)))
    return value



def properties_filter(dic_):
    if type(dic_) == dict:
        dic_r = {}
        for k in dic_:
            if k not in ["parameters", "properties", "description", "type", "example_value", "enum", "items"]:
                continue
            if k == "properties":
                dic_r["properties"] = {_:properties_filter(dic_[k][_]) for _ in dic_[k]}
            elif k == "items":
                dic_r[k] = properties_filter(dic_[k])
            elif k == "type":
                if "|" in dic_[k]:
                    dic_[k] = dic_[k].split("|")[0]
                if dic_[k] == "float":
                    r_ = "number"
                elif dic_[k] in ["list of dictionaries"]:
                    r_ = "object"
                elif dic_[k] in ["int"]:
                    r_ = "integer"
                elif dic_[k] in ["complex_string", "String", "UUID"]:
                    r_ = "string"
                elif "enum" in dic_[k]:
                    try:
                        dic_r["enum"] = json.loads(dic_[k].replace("enum", ""))
                        assert type(dic_r["enum"][0]) == str
                        r_ = "string"
                    except:
                        r_ = "string"
                elif type(dic_[k]) == dict:
                    r_ = "object"
                else:
                    r_ = dic_[k]
                assert r_ in ["string", "integer", "boolean", "array", "object", "number", "enum"], f"Wrong: {r_}"
                dic_r[k] = r_
            elif k == "enum":
                if type(dic_[k]) == dict:
                    enum_ = []
                    for k_ in dic_[k]:
                        assert type(dic_[k][k_]) == list
                        enum_.extend(dic_[k][k_])
                else:
                    enum_ = dic_[k]
                assert type(enum_) == list and all([type(_) in [str, int, float, dict, bool, list] for _ in enum_])
                dic_r[k] = dic_[k]
            else:
                dic_r[k] = dic_[k]
        return dic_r
    else:
        return dic_
    

def functions_uniform(function):
    if type(function) == list and (
        "function" in function[0]
        or "name" in function[0]
    ):
        functions = []
        for function_ in function:
            functions.append(functions_uniform(function_))
        return functions
    function_ = {}
    for key in function:
        if key == "parameters":
            if "properties" not in function[key]:
                function_[key] = {"type": "object", "properties": {}}
            else:
                function_[key] = functions_uniform(function[key])
        elif key == "properties":
            function_[key] = {_:properties_filter(function[key][_]) for _ in function[key]}
        elif key == "function":
            function_[key] = functions_uniform(function[key])
        else:
            function_[key] = function[key]
    return function_


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
    except:
        traceback.print_exc()
    return new_messages