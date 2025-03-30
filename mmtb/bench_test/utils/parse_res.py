import ast


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


def parse_string_to_function(input_str):
    parsed_input = ast.parse(input_str)

    function_name = parsed_input.body[0].value.func.id
    arguments = parsed_input.body[0].value.args
    keywords = parsed_input.body[0].value.keywords

    args_list = []
    for keyword in keywords:
        key = keyword.arg
        value = keyword.value
        value = get_keywords(value)
        args_list.append((key, value))

    return function_name, args_list
