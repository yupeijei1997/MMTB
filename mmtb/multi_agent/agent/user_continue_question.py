import json
import random
import os

from utils import get_all_tool_info, logger


action_type_to_natural_zh = {
    "ST": "你提出的新任务一定需要使用[工具列表]里的一个工具解决。",
    "MT": "你提出的新任务一定需要使用[工具列表]里的多个工具解决。",
    "CQ": "你提出的新任务一定需要使用[工具列表]里的一个工具解决，但是会缺乏必填参数的信息，需要模型反问。",
    "CC": "你提出的新任务模型可以直接使用内部知识回答，不需要使用[工具列表]里任何工具。"
}

action_type_to_natural_en = {
    "ST": "The new task you propose must be solved using one of the tools from the [Tool List].",
    "MT": "The new task you propose must be solved using multiple tools from the [Tool List].",
    "CQ": "The new task you propose must be solved using one of the tools from the [Tool List], but there will be a lack of information for required parameters, necessitating the model to ask follow-up questions.",
    "CC": "The new task can be directly answered by the model using internal knowledge, without needing any tools from the [Tool List]."
}

user_system_prompt_question_for_agent_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体拥有一个Planner、Agent助手，并具备一系列外部工具，可以使用外部工具解决你提出的任务，具体见[工具列表]。
根据上下文对话信息，你已经提出了你的任务，并且超级智能体已经帮你解决了你的任务。
因此，接下来，请你根据上一轮超级智能体的Agent助手的回复继续提出你想让超级智能体帮你解决的新任务。
输出格式参考[用户输出格式]。

[环境信息]="""
{{{env_info}}}
"""

[用户输出格式]="""
用户：根据[要求]，根据上下文对话信息中上一轮以 "Agent助手：" 开头的内容继续提出你的新任务（不要重复这句话）
"""

[要求]="""
1、回复必须以 "用户：" 开头。
{{{example}}}
3、{{{action_type_info}}}
"""

{{{all_tool_required_info}}}

[工具列表]="""
{{{tools}}}
"""'''

user_system_prompt_question_for_agent_template_en = '''Please act as a user who is interacting with a super intelligent agent.
This super intelligent agent has a Planner, an Agent, and a series of external tools that can be used to solve the tasks you propose, as detailed in the [Tool List].
Based on the context of the conversation, you have already proposed your task, and the super intelligent agent has already helped you solve it.
Therefore, next, please continue to propose a new task that you want the super intelligent agent to help you solve, based on the previous response from the super intelligent agent's Agent.
Refer to the [User Output Format] for the output format.

[Environment Information] = """
{{{env_info}}}
"""

[User Output Format] = """
User: Based on the [Requirements], continue to propose your new task based on the content starting with "Agent:" from the previous round of contextual conversation information (do not repeat this sentence).
"""

[Requirements] = """
1. The reply must start with "User:".
{{{example}}}
3. {{{action_type_info}}}
"""

{{{all_tool_required_info}}}

[Tool List] = """
{{{tools}}}
"""'''

reference_example_1_zh = '''
2、你提问时需要对Agent助手的回复内容有所指代，例如：
    对Agent助手的回复进行正数序数指代
    例子：
        用户：高位的股票要下跌时，可以通过哪些迹象发现？（上一轮你提出的任务）
        Agent助手：1、技术分析中的下跌信号 2、市场情绪和交易量分析 ...（上一轮Agent助手的回复）
        用户：详细说一下第二点吧（提出的新任务）
        解释："第二点"是对Agent助手回复中的"2、市场情绪和交易量分析"进行正数序数指代，实际含义为"详细说一下第二点（指代2、市场情绪和交易量分析）吧"
'''
reference_example_1_en = '''
2. When you ask questions, you need to refer to the content of the Agent's response. For example:
   Use positive ordinal numbers to refer to the Agent's responses.
   Example:
   User: What signs can indicate that high-priced stocks are about to fall? (The task you proposed in the previous round)
   Agent: 1. Downward signals in technical analysis 2. Market sentiment and trading volume analysis ... (The Agent's response in the previous round)
   User: Please elaborate on the second point. (Proposed new task)
   Explanation: "Second point" refers to "2. Market sentiment and trading volume analysis" in the Agent's response, meaning "Please elaborate on the second point (referring to 2. Market sentiment and trading volume analysis)."
'''

reference_example_2_zh = '''
2、你提问时需要对Agent助手的回复内容有所指代，例如：
    对Agent助手的回复进行逆数序数指代
    例子：
        用户：趣味篮球方案（上一轮你提出的任务）
        Agent助手：高位放量滞涨 高位大震荡 ... 看跌反扑形态（上一轮Agent助手的回复）
        用户：最后一点细化（提出的新任务）
        解释："最后一点"是对Agent助手回复中的"看跌反扑形态"进行逆数序数指代，实际含义为"最后一点（指代看跌反扑形态）细化"
'''
reference_example_2_en = '''
2. When you ask questions, you need to refer to the content of the Agent's response. For example:
   Use reverse ordinal reference to the Agent's response.
   Example:
   User: Fun basketball plan (the task you proposed in the previous round)
   Agent: High-level volume stagnation, high-level large fluctuations ... Bearish rebound pattern (the response from the Agent in the previous round)
   User: Elaborate on the last point (proposing a new task)
   Explanation: "The last point" is a reverse ordinal reference to the "bearish rebound pattern" in the Agent's response, meaning "elaborate on the last point (referring to the bearish rebound pattern)."
'''

reference_example_3_zh = '''
2、你提问时需要对Agent助手的回复内容有所指代，例如：
    对Agent助手的回复进行疑问代词指代
    例子：
        用户：目前什么行业适合35岁的人进入（上一轮你提出的任务）
        Agent助手：跨境电商行业 风险管理咨询行业 ... 餐饮行业（上一轮Agent助手的回复）
        用户：哪个行业未来机会巨大（提出的新任务）
        解释："哪个行业"是对Agent助手回复中的"跨境电商行业、风险管理咨询行业、餐饮行业"进行疑问代词指代，实际含义为"哪个行业（指代跨境电商行业、风险管理咨询行业、餐饮行业）未来机会巨大"
'''
reference_example_3_en = '''
2. When asking questions, you need to refer to the content of the Agent's response. For example:
    Use interrogative pronouns to refer to the Agent's response.
    Example:
    User: What industries are suitable for a 35-year-old to enter? (Task you proposed in the previous round)
    Agent: Cross-border e-commerce industry, risk management consulting industry, ... catering industry (Agent's response in the previous round)
    User: Which industry has huge future opportunities? (New task proposed)
    Explanation: "Which industry" is an interrogative pronoun referring to "cross-border e-commerce industry, risk management consulting industry, catering industry" in the Agent's response, meaning "which industry (referring to cross-border e-commerce industry, risk management consulting industry, catering industry) has huge future opportunities."
'''

reference_example_4_zh = '''
2、你提问时需要对Agent助手的回复内容有所指代，例如：
    对Agent助手的回复进行简称指代
    例子：
        用户：有哪些国有银行？（上一轮你提出的任务）
        Agent助手：中国银行 中国建设银行 ... （上一轮Agent助手的回复）
        用户：建行电话多少（提出的新任务）
        解释："建行"是对Agent助手回复中的"中国建设银行"进行简称指代，实际含义为"中国建设银行电话多少"
'''
reference_example_4_en = '''
2. When you ask a question, you need to refer to the content of the Agent's response. For example:
   Use abbreviations to refer to the Agent's response.
   Example:
   User: What are the state-owned banks? (The task you asked in the previous round)
   Agent: Bank of China, China Construction Bank... (The response from the Agent in the previous round)
   User: What's the phone number for CCB? (A new task)
   Explanation: "CCB" is an abbreviation referring to "China Construction Bank" from the Agent's response, and it actually means "What is the phone number for China Construction Bank?"
'''

reference_example_5_zh = '''
2、你提问时需要对Agent助手的回复内容有所指代，例如：
    对Agent助手的回复进行内容描述指代
    例子：
        用户：苹果15有哪些型号（上一轮你提出的任务）
        Agent助手：iPhone 15  iPhone 15 Plus iPhone 15 Pro iPhone 15 Pro Max（上一轮Agent助手的回复）
        用户：介绍一下max这款的参数（提出的新任务）
        解释："max这款"是对Agent助手回复中的"iPhone 15 Pro Max"进行内容描述指代，实际含义为"介绍一下max这款（指代iPhone 15 Pro Max）的参数"
'''
reference_example_5_en = '''
2. When you ask questions, you need to refer to the content of the Agent's response. For example:
   Provide a content description reference for the Agent's response.
   Example:
   User: What models are available for the iPhone 15? (The task you proposed in the previous round)
   Agent: iPhone 15, iPhone 15 Plus, iPhone 15 Pro, iPhone 15 Pro Max (The response from the Agent in the previous round)
   User: Tell me about the specs of the Max model. (The new task proposed)
   Explanation: "the Max model" is a content description reference to "iPhone15 Pro Max" in the Agent's response, meaning "Tell me about the specs of the Max model (referring to iPhone 15 Pro Max)."
'''

user_system_prompt_question_for_user_template_zh = '''请你扮演一个用户，你正在和一个超级智能体进行交互。
这个超级智能体拥有一个Planner、Agent助手，并具备一系列外部工具，可以使用外部工具解决你提出的任务，具体见[工具列表]。
根据上下文对话信息，你已经提出了你的任务，并且超级智能体已经帮你解决了你的任务。
接下来，请你根据上一轮的你提出的任务继续提出你想让超级智能体帮你解决的新任务。
输出格式参考[用户输出格式]。

[环境信息]="""
{{{env_info}}}
"""

[用户输出格式]="""
用户：根据[要求]，根据上下文对话信息中上一轮以 "用户：" 开头的内容继续提出你的新任务（不要重复这句话）
"""

[要求]="""
1、回复必须以 "用户：" 开头。
{{{example}}}
3、{{{action_type_info}}}
"""

{{{all_tool_required_info}}}

[工具列表]="""
{{{tools}}}
"""'''
user_system_prompt_question_for_user_template_en = '''Please act as a user who is interacting with a super intelligent agent.
This super intelligent agent has a Planner, an Agent, and a series of external tools that can be used to solve the tasks you propose, as detailed in the[Tool List].
Based on the contextual dialogue information, you have already proposed your task, and the super intelligent agent has helped you solve it.
Next, please continue to propose a new task that you want the super intelligent agent to help you with, based on the task you proposed in the previous round.
The output format should refer to the [User Output Format].

[Environment Information]="""
{{{env_info}}}
"""

[User Output Format]="""
User: Based on the [Requirements], continue to propose your new task based on the content starting with "User:" in the previous round of the contextual dialogue information(do not repeat this sentence).
"""

[Requirements]="""
1. The reply must start with "User:".
{{{example}}}
3. {{{action_type_info}}}
"""

{{{all_tool_required_info}}}

[Tool List]="""
{{{tools}}}
"""'''

omit_example_1_zh = '''
2、你提问时需要对上一轮用户任务的内容有所省略，例如：
    对用户任务的主语进行省略
    例子1：
        用户：打是亲骂是爱（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：最早的出处在哪里？（提出的新任务）
        解释：新任务省略了上一轮用户任务的主语"打是亲骂是爱"，补充省略的主语后，实际含义为"打是亲骂是爱（省略的主语）最早的出处在哪里？"
    例子2：
        用户：朱丽倩是谁的妻子（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：是谁的母亲（提出的新任务）
        解释：新任务省略了上一轮用户任务的主语"朱丽倩"，补充省略的主语后，实际含义为"朱丽倩（省略的主语）是谁的母亲"
'''
omit_example_1_en = '''
2. When you ask a question, you need to omit some content from the previous round of the user's task, for example:
   Omit the subject of the user's task.
   Example 1:
       User: Hitting is affection, scolding is love (the task you proposed in the previous round)
       Agent: xxx (the response from the Agent in the previous round)
       User: Where is the earliest origin? (new task proposed)
       Explanation: The new task omits the subject of the previous round's user task "Hitting is affection, scolding is love." After supplementing the omitted subject, the actual meaning is "Where is the earliest origin of 'Hitting is affection, scolding is love' (omitted subject)?"
   Example 2:
       User: Whose wife is Julie Chen? (the task you proposed in the previous round)
       Agent: xxx (the response from the Agent in the previous round)
       User: Whose mother is she? (new task proposed)
       Explanation: The new task omits the subject of the previous round's user task "Julie Chen." After supplementing the omitted subject, the actual meaning is "Whose mother is Julie Chen (omitted subject)?"
'''

omit_example_2_zh = '''
2、你提问时需要对上一轮用户任务的内容有所省略，例如：
    对用户任务的属性进行省略
    例子：
        用户：北极熊会潜水吗（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：所有的熊都会吗（提出的新任务）
        解释：新任务省略了上一轮用户任务的属性"潜水"，补充省略的属性后，实际含义为"所有的熊都会潜水（省略的属性）吗"
'''
omit_example_2_en = '''
2. When you ask a question, you need to omit some content from the previous round of the user's task, for example:
   Omit attributes of the user's task.
   Example:
   User: Can polar bears dive? (The task you proposed in the previous round)
   Agent: xxx (The response from the AgentAssistant in the previous round)
   User: Can all bears do it? (Proposing a new task)
   Explanation: The new task omits the attribute "dive" from the previous round's user task. After supplementing the omitted attribute, the actual meaning is "Can all bears dive (the omitted attribute)?"
'''

omit_example_3_zh = '''
2、你提问时需要对上一轮用户任务的内容有所省略，例如：
    对用户任务的修饰部分进行省略
    例子：
        用户：国内有哪些好看的悬疑电影？（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：美国电影呢（提出的新任务）
        解释：新任务省略了上一轮用户任务的修饰部分"好看的悬疑"，补充省略的修饰部分后，实际含义为"美国有哪些好看的悬疑电影（省略的修饰部分）呢"
'''
omit_example_3_en = '''
2. When you ask a question, you need to omit some content from the previous round of the user's task. For example:
   Omit the descriptive part of the user's task.
   Example:
   User: What are some good suspense movies in China? (The task you proposed in the previous round)
   Agent: xxx (The response from the Agent in the previous round)
   User: What about American movies? (The new task proposed)
   Explanation: The new task omits the descriptive part "good suspense" from the previous round's user task. After adding the omitted descriptive part, the actual meaning is "What are some good suspense movies in America (omitted descriptive part)?"
'''

omit_example_4_zh = '''
2、你提问时需要对上一轮用户任务的内容有所省略，例如：
    对用户任务的多个句子成分同时进行省略
    例子：
        用户：假如一个倒立摆在月球上，摆动周期是十秒。请问它在地球上的摆动周期是多少？（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：在冥王星上呢？（提出的新任务）
        解释：新任务省略了上一轮用户任务的多个句子成分，包括假设条件："假如一个倒立摆在月球上，摆动周期是十秒。"、问题："请问它在地球上的摆动周期是多少？"，
        补充省略的多个句子成分后，实际含义为"假如一个倒立摆在月球上，摆动周期是十秒。（省略的假设条件）请问"它在冥王星上的摆动周期是多少？（省略的问题）"
'''
omit_example_4_en = '''
2. When you ask a question, you need to omit some content from the previous round of the user's task. For example:
   Omit multiple sentence components of the user's task simultaneously.
   Example:
   User: Suppose an inverted pendulum on the moon has a swing period of ten seconds. What is its swing period on Earth? (The task you proposed in the previous round)
   Agent: xxx (The response from the Agent in the previous round)
   User: What about on Pluto? (New task proposed)
   Explanation: The new task omits multiple sentence components from the previous round of the user's task, including the hypothetical condition: "Suppose an inverted pendulum on the moon has a swing period of ten seconds." and the question: "What is its swing period on Earth?"
   After supplementing the omitted sentence components, the actual meaning is "Suppose an inverted pendulum on the moon has a swing period of ten seconds. (Omitted hypothetical condition) What is its swing period on Pluto? (Omitted question)"
'''

reference_example_6_zh = '''
2、你提问时需要对上一轮用户任务的内容有所指代，例如：
    对用户任务进行人称代词指代
    例子：
        用户：周杰伦的出道时间（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户： 他第一首获奖的歌曲是？（提出的新任务）
        解释："他"是对用户任务中的"周杰伦"进行正数序数指代，实际含义为"他（指代周杰伦）第一首获奖的歌曲是？"
'''
reference_example_6_en = '''
2. When you ask a question, you need to refer to the content of the user's task from the previous round. For example:
   Use pronouns to refer to the user's task.
   Example:
   User: When did Jay Chou debut? (The task you proposed in the previous round)
   Agent:xxx (The response from the Agent in the previous round)
   User: What was his first award-winning song? (The new task proposed)
   Explanation: "He" isa pronoun referring to "Jay Chou" in the user's task, and the actual meaning is "What was his (referring to JayChou) first award-winning song?"
'''

reference_example_7_zh = '''
2、你提问时需要对上一轮用户任务的内容有所指代，例如：
    对用户任务进行指示代词指代
    例子：
        用户：桥对寺门松径小，槛当泉眼石波清。（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户： 这句的出处是什么（提出的新任务）
        解释："这句"是对用户任务中的"桥对寺门松径小，槛当泉眼石波清。"进行指示代词指代，实际含义为"这句（指代桥对寺门松径小，槛当泉眼石波清。）的出处是什么"
'''
reference_example_7_en = '''
2. When you ask a question, you need to refer to the content of the user's task from the previous round. For example:
    Use demonstrative pronouns to refer to the user's task.
    Example:
    User: The bridge faces the temple gate, the pine path is small; the railing is at the spring's eye, the stone waves are clear. (The task you proposed in the previous round)
    Agent: xxx (The response from the Agent in the previous round)
    User: What is the source of this sentence? (Proposing a new task)
    Explanation: "This sentence" is a demonstrative pronoun referring to the user's task "The bridge faces the temple gate, the pine path is small; the railing is at the spring's eye, the stone waves are clear." The actual meaning is "What is the source of this sentence (referring to 'The bridge faces the temple gate, the pine path is small; the railing is at the spring's eye, the stone wave sare clear.')?"
'''

add_example_1_zh = '''
2、你提问时需要对上一轮用户任务新增主语，例如：
    对用户任务新增主语
    例子：
        用户：5天旅游攻略（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：日本（提出的新任务）
        解释："日本"是对用户任务中的"5天旅游攻略"新增了主语日本，实际含义为"日本（新增的主语）5天旅游攻略"
'''
add_example_1_en = '''
When you ask a question, you need to add a subject to the user's task from the previous round. For example:
    Add a subject to the user's task.
    Example:
    User: 5-day travel itinerary (the task you proposed in the previous round)
    Agent: xxx (the response from the Agent in the previous round)
    User: Japan (the new task proposed)
    Explanation: "Japan" adds a subject to the user's task of"5-day travel itinerary," meaning "5-day travel itinerary in Japan" (with "Japan" as the added subject).
'''

add_example_2_zh = '''
2、你提问时需要对上一轮用户任务新增修饰部分，例如：
    对用户任务新增修饰部分
    例子：
        用户：有什么国内看剧的好网站（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：不要付费的（提出的新任务）
        解释："不要付费的"是对用户任务中的"国内看剧的好网站"新增了修饰部分，实际含义为"日本（新增的主语）5天旅游攻略"
'''
add_example_2_en = '''
2. When you ask a question, you need to add a modification to the user's task from the previous round. For example:
    Add a modification to the user's task
    Example:
        User: What are some good websites for watching shows in China? (The task you proposed in the previous round)
        Agent: xxx (The response from the Agent in the previous round)
        User: Free ones (New task proposed)
        Explanation: "Free ones" is an added modification to the user's task of "good websites for watching shows in China," with the actual meaning being "5-day travel guide to Japan (newly added subject)."
'''

modify_example_1_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行主动纠正，例如：
    对用户任务进行纠正修改
    例子：
        用户：北京三日游攻略（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：五日游（提出的新任务）
        解释："五日游"是对用户任务中的"三日游"进行了纠正，实际含义为"北京五日游（纠正修改的内容）攻略"
'''
modify_example_1_en = '''
2. When you ask questions, you need to proactively correct certain parts of the user's task from the previous round. For example:
   Correct and modify the user's task.
   Example:
       User: Three-day tour guide in Beijing (the task you proposed in the previous round)
       Agent: xxx (the response from the Agent in the previous round)
       User: Five-day tour (new task proposed)
       Explanation: "Five-day tour" is a correction of "three-day tour" in the user's task, meaning "Five-day tour guide in Beijing" (the corrected and modified content).
'''

modify_example_2_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行复述修改，例如：
    对用户任务进行复述修改
    例子：
        用户：有什么看剧的好网站（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：国内看剧的好网站（提出的新任务）
        解释："国内"是对用户任务中的"有什么看剧的好网站"进行主动复述，实际含义为"国内有什么看剧的好网站"
'''
modify_example_2_en = '''
2. When you ask a question, you need to paraphrase and modify part of the user's task from the previous round. For example:
   Paraphrase and modify the user's task
   Example:
       User: What are some good websites for watching shows? (The task you proposed in the previous round)
       Agent: xxx (The response from the Agent in the previous round)
       User: Good websites for watching shows in China (Proposing a new task)
       Explanation: "In China" is an active paraphrase of the user's task "What are some good websites for watching shows," with the actual meaning being "What are some good websites for watching shows in China?"
'''

continue_example_1_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行追问，要求重新生成更多内容，例如：
    对用户任务进行追问，要求重新生成更多内容
    例子：
        用户：公司新的办公楼落成，有什么祝福词（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：还有吗？（提出的新任务）
        解释："还有吗？"是对用户任务中的"公司新的办公楼落成，有什么祝福词"进行继续追问，要求重新生成更多内容，实际含义为"公司新的办公楼落成，还有什么祝福词"
'''
continue_example_1_en = '''
2. When you ask questions, you need to follow up on some parts of the user's previous task, requesting the generation of more content.
    Follow up on the user's task and request the generation of more content.
    Example:
        User: What are some congratulatory words for the completion of the company's new office building? (The task you proposed in the previous round)
        Agent: xxx (The response from the Agent in the previous round)
        User: Any more? (The new task proposed)
        Explanation: "Any more?" is a follow-up to the user's task of "What are some congratulatory words for the completion of the company's new office building?" It requests the generation of more content, with the actual meaning being "Are there any more congratulatory words for the completion of the company's new office building?"
'''

continue_example_2_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行追问，要求展开讲述更多内容，例如：
    对用户任务进行追问，要求展开讲述更多内容
    例子：
        用户：有趣且富有创意的中秋节文案（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：可否再丰富一些？（提出的新任务）
        解释："可否再丰富一些？"是对用户任务中的"有趣且富有创意的中秋节文案"进行继续追问，要求展开讲述更多内容，实际含义为"有趣且富有创意的中秋节文案，需要丰富一些"
'''
continue_example_2_en = '''
2. When you ask a question, you need to follow up on some parts of the user's previous task, requesting more detailed information. For example:
   Followup on the user's task and ask for more detailed information.
   Example:
       User: Interesting and creative Mid-Autumn Festival copy writing (the task you proposed in the previous round)
       Agent: xxx (the response from the Agent in the previous round)
       User: Could you elaborate a bit more? (the new task proposed)
       Explanation: "Could you elaborate a bit more?" is a follow-up question on the user's task of "interesting and creative Mid-Autumn Festival copy writing," requesting more detailed information. The actual meaning is "interesting and creative Mid-Autumn Festival copy writing needs to be more enriched."
'''

continue_example_3_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行追问，要求对已有内容简短讲述，例如：
    对用户任务进行追问，要求对已有内容简短讲述
    例子：
        用户：公司新的办公楼落成，有什么祝福词（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：能否简洁一些（提出的新任务）
        解释："能否简洁一些"是对用户任务中的"公司新的办公楼落成，有什么祝福词"进行继续追问，要求对已有内容简短讲述，实际含义为"有趣且富有创意的中秋节文案，需要简洁一些"
'''
continue_example_3_en = '''
2. When you ask a question, you need to follow up on part of the user's task from the previous round, requiring a brief explanation of the existing content. For example:
   Follow up on the user's task, requiring a brief explanation of the existing content.
   Example:
       User: What are some congratulatory words for the completion of the company's new office building? (The task you proposed in the previous round)
       Agent: xxx(The response from the Agent in the previous round)
       User: Can you make it more concise? (The new task proposed)
       Explanation: "Can you make it more concise?" is a follow-up to the user's task of "What are some congratulatory words for the completion of the company's new office building?" It requires a brief explanation of the existing content, with the actual meaning being "Interesting and creative Mid-Autumn Festival copy,needs to be more concise."
'''

continue_example_4_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行追问，要求对已有内容举个示例，例如：
    对用户任务进行追问，要求对已有内容举个示例
    例子：
        用户：mysql 死锁是如何产生的？（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：举个实际例子呢？（提出的新任务）
        解释："举个实际例子呢？"是对用户任务中的"mysql 死锁是如何产生的？"进行继续追问，要求对已有内容举个示例，实际含义为"mysql 死锁是如何产生的？举个实际例子"
'''
continue_example_4_en = '''
2. When you ask a question, you need to follow up on some parts of the user's previous task, requiring an example of the existing content. For example:
   Follow up on the user's task and ask for an example of the existing content.
   Example:
       User: How does a MySQL deadlock occur? (The task you proposed in the previous round)
       Agent: xxx (The response from the Agent in the previous round)
       User: Can you give a practical example? (The new task proposed)
       Explanation: "Can you give a practical example?" is a follow-up question to the user's task "How does a MySQL deadlock occur?" It requires an example of the existing content, with the actual meaning being "How does a MySQL deadlock occur? Can you give a practical example?"
'''

continue_example_5_zh = '''
2、你提问时需要对上一轮用户任务的部分内容进行追问，要求对已有内容接续生成，例如：
    对用户任务进行追问，要求对已有内容接续生成
    例子：
        用户：给我出师表全文（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：继续（提出的新任务）
        解释："继续"是对用户任务中的"给我出师表全文"进行继续追问，要求对已有内容接续生成，实际含义为"给我出师表全文，继续生成"
'''
continue_example_5_en = '''
2. When you ask questions, you need to follow up on some parts of the user's previous task, requiring continuation of the existing content. For example:
    Follow up on the user's task, requiring continuation of the existing content.
    Example:
        User: Give me the full text of "Chu Shi Biao" (the task you proposed in the previous round)
        Agent: xxx (the response from the Agent in the previous round)
        User: Continue (the new task proposed)
        Explanation: "Continue" is a follow-up to the user's task of "Give me the full text of 'Chu Shi Biao'," requiring continuation of the existing content. The actual meaning is "Give me the full text of 'Chu Shi Biao,' continue generating."
'''

switch_example_1_zh = '''
2、你提问时需要根据上一轮用户任务进行相邻对话轮次相似话题切换，例如：
    对用户任务进行相似话题切换
    例子：
        用户：哪些水果含有丰富的维生素C？（上一轮你提出的任务）
        Agent助手：xxx（上一轮Agent助手的回复）
        用户：维生素C对身体有哪些好处？（提出的新任务）
        解释：上一轮用户任务是询问"哪些水果含有丰富的维生素C？"，这一轮提出的新任务是询问"维生素C对身体有哪些好处？"，两轮用户任务都涉及维生素C相关的问题，属于相邻对话轮次相似话题切换
'''
switch_example_1_en = '''
2. When you ask a question, you need to switch to a similar topic in adjacent dialogue rounds based on the user's previous task. For example:
   Switching to a similar topic for the user's task
   Example:
       User: Which fruits are rich in vitamin C? (The task you proposed in the previous round)
       Agent: xxx (The response from the Agent in the previous round)
       User: What are the benefits of vitamin C for the body? (The new task proposed)
       Explanation: In the previous round, the user's task was to ask "Which fruits are rich in vitamin C?" In this round, the new task is to ask "What are the benefits of vitamin C for the body?" Both rounds of user tasks involve questions related to vitamin C, which constitutes a switch to a similar topic in adjacent dialogue rounds.
'''

switch_example_2_zh = '''
2、你提问时需要根据前几轮用户任务进行间隔轮次话题循环切换，例如：
    对用户任务进行话题循环切换
    例子：
        用户：十二生肖为什么没有猫？（第一轮你提出的任务）
        Agent助手：xxx（第一轮Agent助手的回复）
        用户：中国最大的原始森林区是？（第二轮你提出的任务）
        Agent助手：xxx（第二轮Agent助手的回复）
        用户：为什么男装纽扣在右，而女装纽扣在左呢？（第三轮你提出的任务）
        Agent助手：xxx（第三轮Agent助手的回复）
        用户：猫原产于哪里（第四轮提出的新任务）
        解释：第四轮提出的新任务是询问"猫原产于哪里"，这与第一轮询问"十二生肖为什么没有猫？"的话题又重新关联在了一起，属于间隔对话轮次话题循环切换
'''
switch_example_2_en = '''
2. When asking questions, you need to alternate topics in a cyclic manner based on the user's tasks from previous rounds. For example:
   Perform topic cycling based on the user's tasks.
   Example:
       User: Why is there no cat in the Chinese zodiac? (Task you proposed in the first round)
       Agent: xxx (Agent's response in the first round)
       User: Where is the largest area of virgin forest in China? (Task you proposed in the second round)
       Agent: xxx (Agent's response in the second round)
       User: Why are buttons on the right for men's clothing andon the left for women's clothing? (Task you proposed in the third round)
       Agent: xxx (Agent's response in the third round)
       User: Where do cats originate from? (New task proposed in the fourth round)
       Explanation: The new task proposed in the fourth round is asking "Where do cats originate from?" which is related back to the first round's question "Why is there no cat in the Chinese zodiac?" This is an example of alternating topics in a cyclic manner across different rounds of conversation.
'''

user_system_prompt_template_list_zh = [
    user_system_prompt_question_for_agent_template_zh.replace("{{{example}}}", reference_example_1_zh),
    user_system_prompt_question_for_agent_template_zh.replace("{{{example}}}", reference_example_2_zh),
    user_system_prompt_question_for_agent_template_zh.replace("{{{example}}}", reference_example_3_zh),
    user_system_prompt_question_for_agent_template_zh.replace("{{{example}}}", reference_example_4_zh),
    user_system_prompt_question_for_agent_template_zh.replace("{{{example}}}", reference_example_5_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", omit_example_1_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", omit_example_2_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", omit_example_3_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", omit_example_4_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", reference_example_6_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", reference_example_7_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", add_example_1_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", add_example_2_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", modify_example_1_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", modify_example_2_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", continue_example_1_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", continue_example_2_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", continue_example_3_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", continue_example_4_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", continue_example_5_zh),

    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", switch_example_1_zh),
    user_system_prompt_question_for_user_template_zh.replace("{{{example}}}", switch_example_2_zh)
]


user_system_prompt_template_list_en = [
    user_system_prompt_question_for_agent_template_en.replace("{{{example}}}", reference_example_1_en),
    user_system_prompt_question_for_agent_template_en.replace("{{{example}}}", reference_example_2_en),
    user_system_prompt_question_for_agent_template_en.replace("{{{example}}}", reference_example_3_en),
    user_system_prompt_question_for_agent_template_en.replace("{{{example}}}", reference_example_4_en),
    user_system_prompt_question_for_agent_template_en.replace("{{{example}}}", reference_example_5_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", omit_example_1_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", omit_example_2_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", omit_example_3_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", omit_example_4_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", reference_example_6_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", reference_example_7_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", add_example_1_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", add_example_2_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", modify_example_1_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", modify_example_2_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", continue_example_1_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", continue_example_2_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", continue_example_3_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", continue_example_4_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", continue_example_5_en),

    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", switch_example_1_en),
    user_system_prompt_question_for_user_template_en.replace("{{{example}}}", switch_example_2_en)
]

def user_continue_question(messages, tools, env_info, request_func, node):
    language = os.getenv("LANGUAGE")
    if language == "zh":
        user_system_prompt_template = random.choice(user_system_prompt_template_list_zh)
        action_type_info = action_type_to_natural_zh[node]
    else:
        user_system_prompt_template = random.choice(user_system_prompt_template_list_en)
        action_type_info = action_type_to_natural_en[node]

    all_tool_name, all_tool_required_info = get_all_tool_info(tools)
    user_system_prompt = user_system_prompt_template.replace("{{{tools}}}", json.dumps(tools, ensure_ascii=False, indent=4)) \
                                                    .replace("{{{env_info}}}", env_info) \
                                                    .replace("{{{all_tool_required_info}}}", all_tool_required_info) \
                                                    .replace("{{{action_type_info}}}", action_type_info)
    # print(user_system_prompt)
    messages_new = [
        {
            "role": "system",
            "content": user_system_prompt
        }
    ]
    messages_new.extend(messages)
    res = request_func(messages_new)
    logger.info(f"user_continue_question:\n{res}\n")
    fetch_data = {"task": "user_continue_question", "tools": tools, "env_info": env_info, "messages": messages_new, "answer": res}
    return res, fetch_data
