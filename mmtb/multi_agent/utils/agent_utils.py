import json
import os
import random
import re


def parse_answer(planner_res):
    pattern = "```json(.+?)```"
    planner_res = re.findall(pattern, planner_res, re.S)[0]
    planner_res_obj = json.loads(planner_res)
    return planner_res_obj


def random_select_answer(user_tasks):
    user_tasks = user_tasks.replace("```json\n", "").replace("\n```", "")
    user_tasks = json.loads(user_tasks)
    task_keys = list(user_tasks.keys())
    task_key = random.choice(task_keys)
    user_task = user_tasks[task_key]
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_task = "用户：" + user_task
    else:
        user_task = "User: " + user_task

    return user_task


def random_select_answer_cot(user_tasks):
    user_tasks = user_tasks.replace("```json\n", "").replace("\n```", "")
    user_tasks = json.loads(user_tasks)
    task_keys = list(user_tasks.keys())
    task_key = random.choice(task_keys)
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_task = user_tasks[task_key]["任务描述"]
        user_task = "用户：" + user_task
    else:
        user_task = user_tasks[task_key]["Task Description"]
        user_task = "User: " + user_task
    return user_task


def get_all_tool_info(tools):
    all_tool_name = []
    all_tool_required_info = []
    for tool in tools:
        tool_name = tool["function"]["name"]
        all_tool_name.append(tool_name)
        tool_required = tool["function"]["parameters"]["required"]
        tool_required = "[" + ", ".join(tool_required) + "]"
        tool_all_properties = list(tool["function"]["parameters"]["properties"].keys())
        tool_no_required = []
        for property in tool_all_properties:
            if property not in tool_required:
                tool_no_required.append(property)
        tool_no_required = "[" + ", ".join(tool_no_required) + "]"
        language = os.getenv("LANGUAGE")
        if language == "zh":
            tool_required_info = f"工具{tool_name}的必填参数为：{tool_required}，非必填参数为：{tool_no_required}"
        else:
            tool_required_info = f"The required parameters for the tool {tool_name} are: {tool_required}, and the optional parameters are: {tool_no_required}."
        all_tool_required_info.append(tool_required_info)
    all_tool_name = ", ".join(all_tool_name)
    all_tool_required_info = "\n".join(all_tool_required_info)
    return all_tool_name, all_tool_required_info


def get_all_tool_info_for_checker(tools):
    all_tool_name = []
    all_tool_name_properties_name = {}
    all_tool_name_required = {}
    for tool in tools:
        tool_name = tool["function"]["name"]
        all_tool_name.append(tool_name)
        tool_properties = list(tool["function"]["parameters"]["properties"].keys())
        all_tool_name_properties_name[tool_name] = tool_properties

        tool_required = tool["function"]["parameters"]["required"]
        all_tool_name_required[tool_name] = tool_required

    return all_tool_name, all_tool_name_properties_name, all_tool_name_required


if __name__ == "__main__":
    tools = [
            {
                "type": "function",
                "function": {
                    "name": "getGeocode",
                    "description": "根据新加坡的地址获取地理编码信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "需要查询的新加坡地址。"
                            },
                            "returnGeom": {
                                "type": "boolean",
                                "description": "是否返回地理坐标信息，默认为false。"
                            }
                        },
                        "required": [
                            "address"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "getReverseGeocode",
                    "description": "根据地理坐标获取新加坡的地址信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {
                                "type": "float",
                                "description": "纬度值。"
                            },
                            "longitude": {
                                "type": "float",
                                "description": "经度值。"
                            },
                            "buffer": {
                                "type": "integer",
                                "description": "搜索半径范围，默认为50米。"
                            }
                        },
                        "required": [
                            "latitude",
                            "longitude"
                        ]
                    }
                }
            },
        {
            "type": "function",
            "function": {
                "name": "getLocationBasedServices",
                "description": "获取新加坡基于位置的服务信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "服务类别。"
                        },
                        "location": {
                            "type": "object",
                            "description": "位置坐标对象。",
                            "properties": {
                                "latitude": {
                                    "type": "float",
                                    "description": "纬度值。"
                                },
                                "longitude": {
                                    "type": "float",
                                    "description": "经度值。"
                                }
                            }
                        },
                        "radius": {
                            "type": "integer",
                            "description": "搜索半径，默认为500米。"
                        }
                    },
                    "required": [
                        "category",
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "getRoutePlanning",
                "description": "提供新加坡的路线规划服务。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "startPoint": {
                            "type": "object",
                            "description": "起点坐标。",
                            "properties": {
                                "latitude": {
                                    "type": "float",
                                    "description": "起点纬度。"
                                },
                                "longitude": {
                                    "type": "float",
                                    "description": "起点经度。"
                                }
                            }
                        },
                        "endPoint": {
                            "type": "object",
                            "description": "终点坐标。",
                            "properties": {
                                "latitude": {
                                    "type": "float",
                                    "description": "终点纬度。"
                                },
                                "longitude": {
                                    "type": "float",
                                    "description": "终点经度。"
                                }
                            }
                        },
                        "mode": {
                            "type": "string",
                            "description": "出行模式，默认为'driving'。",
                            "enum": [
                                "driving",
                                "walking",
                                "cycling"
                            ]
                        }
                    },
                    "required": [
                        "startPoint",
                        "endPoint"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "getVisualization",
                "description": "支持新加坡数据的可视化展示。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {
                            "type": "string",
                            "description": "需要展示的数据层。"
                        },
                        "theme": {
                            "type": "string",
                            "description": "可视化主题，默认为'standard'。"
                        },
                        "zoomLevel": {
                            "type": "integer",
                            "description": "地图缩放级别，默认为10。"
                        }
                    },
                    "required": [
                        "layer"
                    ]
                }
            }
        }
    ]
    all_tool_name, all_tool_required_info = get_all_tool_info(tools)
    print(f"all_tool_name: {all_tool_name}")
    print(f"all_tool_required_info: \n{all_tool_required_info}")
