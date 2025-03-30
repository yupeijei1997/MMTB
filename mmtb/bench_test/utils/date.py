import time
import random

from datetime import datetime


def get_random_date(a1=(2024, 1, 1, 0, 0, 0, 0, 0, 0), a2=(2024, 12, 31, 23, 59, 59, 0, 0, 0)):
    start = time.mktime(a1)  # 生成开始时间戳
    end = time.mktime(a2)  # 生成结束时间戳

    t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(t)  # 将时间戳生成时间元组
    date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
    return date


def get_current_date():
    current_time = time.time()  # 获取当前时间戳
    current_date_tuple = time.localtime(current_time)  # 将时间戳生成时间元组
    current_date = time.strftime("%Y-%m-%d %H:%M:%S", current_date_tuple)  # 将时间元组转成格式化字符串
    return current_date


def add_weekday_date(date):
    if "星期" in date:
        return date
    date = date.replace("当前时间：", "").replace("环境：", "")
    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    weekday_num = date_obj.weekday()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[weekday_num]
    date = "当前时间：" + date + " " + weekday
    return date


def get_current_date_with_weekday():
    return add_weekday_date(get_current_date())