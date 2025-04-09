import requests

from datetime import datetime


class SimulateMultiTurnMessages:
    def __init__(self, model_url, is_english):
        self.model_url = model_url
        self.is_english = is_english
        self.model_messages = []
        self.timeout = 90
        self.add_date = True

    def preprocess_to_simple(self, messages):
        pass

    def post_process_tool_call(self, answer):
        pass

    def add_weekday_date(self, date):
        date = date.replace("当前时间：", "").replace("环境：", "")
        date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        weekday_num = date_obj.weekday()
        if self.is_english:
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        else:
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday = weekdays[weekday_num]
        date = date + " " + weekday
        return date

    def add_date_to_message(self, message, env_info=None):
        if env_info is not None and self.add_date:
            system_content = message[0]["content"] if message[0]["role"] == "system" else ""
            if self.is_english:
                system_content = system_content[:system_content.rfind("Current Date:")] + "\n\nCurrent Date:" + self.add_weekday_date(env_info)
            else:
                system_content = system_content[:system_content.rfind("当前日期：")] + "当前日期：" + self.add_weekday_date(env_info)
            if message[0]["role"] == "system":
                message[0]["content"] = system_content.strip()
            else:
                message.insert(0, {"role": "system", "content": system_content.strip()})
            return message
        else:
            return message

    def request_funcall(self, messages, tools, env_info=None):
        url = self.model_url
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": self.add_date_to_message(self.preprocess_to_simple(messages), env_info),
            "tools": tools,
            "date": self.add_weekday_date(env_info)
        }

        text = None
        tool_calls = None
        try:
            response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                text, tool_calls = self.post_process_tool_call(answer)
        except Exception as e:
            print(f"error: {e}")
            text = None
            tool_calls = None

        return text, tool_calls
