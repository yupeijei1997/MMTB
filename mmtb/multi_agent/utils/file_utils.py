import json


def read_json_file_to_list(input_file):
    result = []
    with open(input_file) as fin:
        for line in fin:
            obj = json.loads(line)
            result.append(obj)
    return result


def write_json_data_to_file(data_list, output_file):
    fout = open(output_file, "w")
    for data in data_list:
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
    fout.close()
