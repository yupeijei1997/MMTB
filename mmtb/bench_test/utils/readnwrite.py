import json
import pandas as pd

from glob import glob


def read_json_file_to_list(input_file):
    result = []
    with open(input_file) as fin:
        for line in fin:
            obj = json.loads(line)
            result.append(obj)
    return result


def read_file_to_json(path, skip_path=None):
    data = []
    files = glob(path)
    print("Read files:")
    for file_ in files:
        if skip_path is not None and (
                file_ == skip_path
                or file_ in skip_path
        ):
            continue
        with open(file_, "r") as f:
            tmps = [json.loads(_) for _ in f.readlines()]
            print(f"{file_}: {len(tmps)}")
            data += tmps
    return data


def write_json_to_file(data, path, func=None, print_f=True):
    with open(path, "w") as f:
        for item in data:
            if func != None:
                item = func(item)
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    if print_f:
        print(f"Write {len(data)} items to {path}\nSamples: {json.dumps(item, ensure_ascii=False)}")
    else:
        print(f"Write {len(data)} items to {path}")


def read_csv_to_dict_list(file_path):
    # 使用pandas读取csv文件
    df = pd.read_csv(file_path, keep_default_na=False, na_values=[''])

    # 重命名"Unnamed: {num}"列
    all_none_from = None
    for col in df.columns:
        if col.startswith("Unnamed") and df[col].isnull().all():
            all_none_from = df.columns.get_loc(col)
            break

    # 如果存在这样的列，则删除这些列
    if all_none_from is not None:
        df = df.iloc[:, :all_none_from]

    df.rename(columns=lambda x: f"key{x.split(': ')[1]}" if x.startswith("Unnamed") else x, inplace=True)

    # 将DataFrame转换为字典列表
    dict_list = df.to_dict('records')

    # 过滤掉所有值都是None的字典
    filtered_dict_list = []
    for row in dict_list:
        # 将空字符串替换为None，并检查是否所有值都是None
        all_none = True
        for key in list(row.keys()):
            if pd.isna(row[key]):
                row[key] = None
            else:
                all_none = False
        # 如果不是所有值都是None，添加到结果列表中
        if not all_none:
            filtered_dict_list.append(row)

    print(f"Read file: {file_path}\ndata length:{len(filtered_dict_list)}\nkeys:{filtered_dict_list[0].keys()}")
    return filtered_dict_list


def write_list_of_list_to_csv(list_of_list, csv_file_name):
    # 检查list_of_list是否至少有两个元素（列名和至少一行数据）
    if len(list_of_list) < 2:
        raise ValueError("List of list must contain at least one row of data along with column names.")
    assert all([len(_) == len(list_of_list[0]) for _ in list_of_list])

    # 第一个元素是列名
    column_names = list_of_list[0]

    # 剩余的元素是数据行
    data_rows = list_of_list[1:]

    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)

    # 导出到CSV文件
    df.to_csv(csv_file_name, index=False)  # index=False表示不导出行索引
    print(f"Write data to {csv_file_name}\nSamples: {len(list_of_list)}")
