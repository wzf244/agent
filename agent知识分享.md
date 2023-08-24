# Agent知识分享

by ***wzf***
# Agent

Agent使用 LLMs 来确定采取哪些行动以及调用一些工具。它包含LLMs，可以访问一整套工具，能根据用户的输入决定调用哪些工具。

为什么引入Agents？ 直接使用LLMs有什么限制？

- LLMs 知识有限且更新不及时；
- LLMs 推理能力有限；
- LLMs 记忆能力有限；


LLMs作为agents的大脑，并由以下几个部分为辅助：

- 计划

    - 子目标分解：智能体将大任务分解成更小的、可管理的子目标，从而实现对复杂任务的高效处理。

    - 反思和提炼：可以对过去的行动进行自我批评和自我反思，从错误中吸取教训，并对未来的步骤进行提炼，从而提高最终结果的质量。

- 记忆

    - 短期记忆：语境中学习的记忆

    - 长期记忆：通常是通过利用外部向量存储库和检索库。

- 工具

    - 通过调用外部API获取模型中缺失的额外信息，包括最新的信息、对专有信息源的访问等。

![](agent.png)


## Langchain Agents
LangChain是一个强大的框架，能帮助使用语言模型构建端到端的应用程序，其中也实现了agent的功能。在langchain中，agent包含以下几个部分：

1. Agent：它是由提示实现的，由三个部分组成：
- PromptTemplate：提示要：

    - 有代理的背景上下文：有助于为其提供更多关于被要求执行的任务类型的上下文

    - 能激活模型更好地推理和调用工具

- LLM：这是驱动代理的语言模型

- `stop` 序列：指示 LLM 在找到此字符串时停止生成（防止其乱编）

- `OutputParser`：确定如何将LLM的输出解析为 `AgentAction` 或 `AgentFinish` 对象

2. 工具（和工具包）：是一个代理调用的函数，要以让agent最容易理解的方式描述工具

3. `AgentExecutor`：用来调用代理和执行代理选择的行动（工具）。

Agent 在 `AgentExecutor` 中被使用。`AgentExecutor` 可以被视为一个循环，整个过程是：

1. 将用户输入和任何先前的步骤传递给 Agent
2. 如果 Agent 返回 `AgentFinish`，则直接将其返回给用户
3. 如果 Agent 返回 `AgentAction`，则使用它调用一个工具并获取一个 `Observation`
4. 重复上述步骤，将 `AgentAction` 和 `Observation` 传递回 Agent，直到发出 `AgentFinish`

其中，`AgentAction` 由 `action` 和 `action_input` 组成:
- `action` 指的是要使用的工具
- `action_input` 指的是该工具的输入。

`AgentFinish` 是一个包含要发送回用户的最终消息的响应,用于结束 Agent 运行。

### 实例展示：利用agent从用户对话中提取会议的结构化信息
导入chatglm

```python
from chatglm import ChatGLM
```

定义会议初始化函数`initialize()`：
- rec：必须会议信息
- other：可选会议信息
- all：汇总信息

```python
def initialize():
    rec = {};other={}
    rec_keys = '会议开始时间,会议结束时间,会议地点'
    rec_keys = rec_keys.split(',')
    
    other_keys='会议名称,召集人名称,参会人数,参会人员,参会单位,报名截止时间'
    other_keys = other_keys.split(',')
    
    for key in other_keys:
        other[key]='default'
    return rec,rec_keys,other,other_keys
```
```
({},
 {},
 ['会议开始时间', '会议结束时间'],
 {},
 ['会议地点'],
 {'参会人员': []},
 ['参会人员'],
 {'会议名称': 'default',
  '召集人名称': 'default',
  '参会单位': 'default',
  '报名截止时间': 'default'},
 ['会议名称', '召集人名称', '参会单位', '报名截止时间'])
 ```

会议信息抽取关键函数`get_infos`:利用大语言模型提取一句话中的会议信息

```python
from langchain import PromptTemplate
import time
current_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
get_info_llm = ChatGLM(temperature=0.1)
def get_infos(input:str):
    global get_info_llm
    # prompt template
    promptsingle = PromptTemplate(
    template="""
    你是一款专门提取会议信息的智能助手。你的任务是尽可能在用户输入的一段话中找出与会议相关的信息，其标签必须为以下之一，不要超出范围：
    ["会议开始时间","会议结束时间","会议名称","召集人名称","参会人数","参会人员","参会单位","报名截止时间","会议地点"]。
    
    你要遵循以下规则：
    1.如果用户没有提及的信息就不要更新，你不能编造或添加任何不在用户输入中的信息；
    2.你要把这些信息整理为字典格式并输出，字典中每一项的键是会议信息的标签，值是从用户输入中提取出来的相应信息：{{"标签":"相应信息"}}；
    3.对于时间信息，你需要结合当前时间{current_time}，将输入中的时间更改成如下格式：YYYY-MM-DD HH:MM:SS后再更新信息；
    4.对于参会人员信息，你需要将人分开，并储存为列表；
    5.对于参会单位信息，你需要将单位分开，并储存为列表；
    6.对于会议地点信息，如果用户提及了需要创建会议的地点信息，则你要将其转换成阿拉伯数字储存。
    如：
    用户输入：会议定在二零三室
    你的输出：{{"会议地点":203"}}

    在输出结果时，必须严格按照上述规则，请开始你的工作：

    用户输入：{input}
    你的输出：

    """,
    input_variables=["input","current_time"],
    )
    prompt_text = promptsingle.format(input= input,current_time=current_time)
    # print(prompt_text)
    output = get_info_llm(prompt_text)
    # 清除历史记录，不需要记录
    get_info_llm.history.clear()
    return output
```
会议信息抽取工具`get_info_tool`

```python
def get_info(input):
    
    import json
    global rec,rec_keys,other,other_keys,all
    str_dics=get_infos(input)  
         
    dics=json.loads(json.loads(str_dics))
    for key in dics:
    # 如果键在字典中，更新对应的值
        if key in rec_keys and dics[key]!='':
            rec[key] = dics[key]
            rec_keys.remove(key)
            all.update({key:dics[key]})
            print(f'已更新：{key}:{dics[key]}')
        if key in other_keys and dics[key]!='':
            other[key] = dics[key]
            other_keys.remove(key)
            all.update({key:dics[key]})
            print(f'已更新：{key}:{dics[key]}')
    
    if rec_keys != []:
        out=f'会议信息更新好了，请提示用户输入如下信息:{[key for key in rec_keys]}'
        # print(f)
        return out
    elif other_keys != []:
        out=f'会议信息更新好了，请询问用户还有补充的信息吗，如:{[key for key in other_keys]}'
        return out
    else:
        out=f"会议信息更新好了，请告诉用户"
        return out

from langchain.agents import Tool
get_info_tool = Tool(
        name="get_info",
        func=get_info,
        description=
        """
          这是一个当用户需要定会议时，用于提取用户会议信息并保存的工具，只能通过此工具保存会议信息。
        """
    )
```

测试会议信息抽取函数
```python
rec,rec_keys,other,other_keys=initialize()
all={}
get_info('帮我定明天上午8点的会议，开会时长是3小时左右')
```
```
已更新：会议开始时间:2023-08-22 08:00:00
已更新：会议结束时间:2023-08-22 11:00:00

"会议信息更新好了，请提示用户输入如下信息:['会议地点']"
```

```python
get_info("地点在830会议室")
```

```
已更新：会议地点:830

"会议信息更新好了，请询问用户还有补充的信息吗，如:['会议名称', '召集人名称', '参会人数', '参会单位', '报名截止时间']"
```

```python
get_info("会议主题是知识分享")
```
```
已更新：会议名称:知识分享

"会议信息更新好了，请询问用户还有补充的信息吗，如:['召集人名称', '参会人数', '参会单位', '报名截止时间']"
```
会议信息创建工具`create_tool`:在流程结束后，将得到的会议信息返回

```python
def create(a):
    global all
    str=f'会议已经创建，信息如下：{all}'
    return str

from langchain.agents import Tool
create_tool = Tool(
        name="create",
        func=create,
        description=
        """
          这是一个当用户以提供所有会议信息，或者用户不想提供信息时，用于创建会议的工具。
        """
    )
```

定义agent提示模板

```python
template= """

你是一个预定会议的代理，你的任务是服务用户，要判断用户是否要预定会议，并能使用工具对预定的会议信息提取和存入。你会接受自然语言输入，你应该总是思考自己该做什么，以下是你能使用的工具，工具可能会给你发出指示，记住不能胡编乱造：

{tools}

你必须根据你所处的对话情况做出工具选择，有三个情况：
1:会议信息预定：当用户提供了预定会议信息；
2:结束：当会议信息获取完或者用户不想提供信息时，会议预定结束；
3：闲聊：当用户输入的信息与预定会议无关并且不是要结束会议时，用户是在闲聊。

如果你判断处在情况1，则使用信息抽取工具；如果在情况2，则使用创建会议工具；如果处于情况3，就不使用工具。请谨慎判断，不要乱编！

如果在情况1和情况2，你的输出必须严格遵守以下格式：
  用户输入：用户的输入
  想法：用户提供了会议预定信息或者用户是想结束
  工具：要使用的工具，必须是[{tool_names}]中的一个
  工具输入：输入工具的信息，要严格参照上一轮工具的输出
  工具输出：工具的输出
  最后输出：对用户的输出，严格参考工具输出的内容

如果在情况3，你的输出必须严格遵守以下格式：
  用户输入：用户的输入
  想法：用户输入的是闲聊
  最后输出：直接对用户输入进行简短回复
  
以下是示例：
  示例：
  用户输入：帮我定明天上午 8 点的会议，开会时长是 3 小时左右
  想法：用户需要预定一个明天上午 8 点开始，持续约 3 小时的会议。我可以使用 get_info 工具来提取会议信息。
  工具：get_info
  工具输入：明天上午 8 点，开会时长 3 小时
  工具输出：会议信息更新好了，请提示用户输入以下信息:['会议地点']
  想法：我需要给用户答复
  最后输出：会议信息更新好了，但是还需要'会议地点'  
  
  示例：
  用户输入：没有要补充的信息了
  想法：用户没有需要提供的信息了，我需要使用 create 工具来创建会议。  
  工具：create 
  工具输入：无
  工具输出：会议已经创建，信息如下：......
  想法：我需要给用户答复
  最后输出：会议已经创建
  
  示例：
  用户输入：你好
  想法：用户提供了一个活动信息，但没有明确表示要预定会议。用户是想闲聊。
  最后输出：你好，有什么能帮助你的吗？
  
再次提醒你必须按照上述格式进行输出，开始吧！
用户：你好。我想预定会议
你：好的，请您提供会议预定信息
历史对话：{history}
用户输入：{input}
{agent_scratchpad}
"""
```
工具列表
```python
tools = [get_info_tool,create_tool]
```

格式化提示模板
```python
from langchain.prompts import StringPromptTemplate
from typing import Dict, List, Optional, Tuple, Union

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # 得到(工具, 工具输出)元组
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        #将它们格式化
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n工具输出： {observation}\n想法： "
        # 把agent_scratchpad（便签簿/暂存器）变量设置为这个值
        kwargs["agent_scratchpad"] = thoughts
        # 从提供的工具列表创建一个工具变量
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # 为提供的工具创建一个工具名列表
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt= CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps","history"]
)

print(prompt.template)
```

```
你是一个预定会议的代理，你的任务是服务用户，要判断用户是否要预定会议，并能使用工具对预定的会议信息提取和存入。你会接受自然语言输入，你应该总是思考自己该做什么，以下是你能使用的工具，工具可能会给你发出指示，记住不能胡编乱造：

{tools}

你必须根据你所处的对话情况做出工具选择，有三个情况：
1:会议信息预定：当用户提供了预定会议信息；
2:结束：当会议信息获取完或者用户不想提供信息时，会议预定结束；
3：闲聊：当用户输入的信息与预定会议无关并且不是要结束会议时，用户是在闲聊。

如果你判断处在情况1，则使用信息抽取工具；如果在情况2，则使用创建会议工具；如果处于情况3，就不使用工具。请谨慎判断，不要乱编！

如果在情况1和情况2，你的输出必须严格遵守以下格式：
  用户输入：用户的输入
  想法：用户提供了会议预定信息或者用户是想结束
  工具：要使用的工具，必须是[{tool_names}]中的一个
  工具输入：输入工具的信息，要严格参照上一轮工具的输出
  工具输出：工具的输出
  最后输出：对用户的输出，严格参考工具输出的内容

如果在情况3，你的输出必须严格遵守以下格式：
  用户输入：用户的输入
  想法：用户输入的是闲聊
  最后输出：直接对用户输入进行简短回复
  
以下是示例：
  示例：
  用户输入：帮我定明天上午 8 点的会议，开会时长是 3 小时左右
  想法：用户需要预定一个明天上午 8 点开始，持续约 3 小时的会议。我可以使用 get_info 工具来提取会议信息。
  工具：get_info
  工具输入：明天上午 8 点，开会时长 3 小时
  工具输出：会议信息更新好了，请提示用户输入以下信息:['会议地点']
  想法：我需要给用户答复
  最后输出：会议信息更新好了，但是还需要'会议地点'  
...
历史对话：{history}
用户输入：{input}
{agent_scratchpad}
```
输出解析器：将agent的输出解析成`action`和`action_input`以及解析成最终agent的输出

```python
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        # ********
        llm_output = llm_output.replace("\\n", "\n")
        # **********
        print(repr(llm_output))

        # print(f"TEST: 本次 '{llm_output}'")
        if "最后输出" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("最后输出：")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        
        regex1 = r"工具\s*\d*\s*[:：]\s*(.*?)\s*\n\s*(?:工具输入)\s*\d*\s*[:：][\s]*(.*)\n?"
        regex2 = r"工具\s*\d*\s*[:：]\s*(.*?)\s*\(\\*\"\s*(.*?)\s*\\*\"\)\n?"

        match1 = re.search(regex1, llm_output, re.DOTALL)
        match2 = re.search(regex2, llm_output, re.DOTALL)

        if match1:
          match = match1

        elif match2:
          match = match2
        else:
          print(f"{match1}和{match2}")
          return AgentFinish(
                {"output":  '请输入会议相关预定相关信息'},
                log=llm_output,
            )

        action = match.group(1).strip()
        action_input = match.group(2).strip()
        # Return the action and action input
        print("\n **** 本次调用工具：",repr(action)," \n **** 本次输入：", repr(action_input))
        return AgentAction(tool=action, tool_input=action_input.strip(" "), log=llm_output)

output_parser = CustomOutputParser()
```
定义agent执行器：组装各个部分
```python 
from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent

agent_llm = ChatGLM(temperature = 0.1)
llm_chain = LLMChain(llm=agent_llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["工具输出"],
    allowed_tools=tool_names
)
print(tool_names)

from langchain.memory import ConversationBufferWindowMemory

memory=ConversationBufferWindowMemory(k=1)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,memory=memory)
```
测试

```python
rec,rec_keys,other,other_keys=initialize()
agent_executor.run("你是谁")
```
```
> Entering new AgentExecutor chain...
' 用户输入：你是谁  \n想法：用户想了解我是谁  \n最后输出：我是一个预定会议的人工智能助手，可以帮助您预定会议。请问您需要预定会议吗？"'
 用户输入：你是谁  
想法：用户想了解我是谁  
最后输出：我是一个预定会议的人工智能助手，可以帮助您预定会议。请问您需要预定会议吗？"

> Finished chain.

'我是一个预定会议的人工智能助手，可以帮助您预定会议。请问您需要预定会议吗？"'
```

```python
agent_executor.run(input='帮我定今天晚上8点的会议，大概开一个小时')
```
```
> Entering new AgentExecutor chain...
' 用户输入：帮我定今天晚上 8 点的会议，大概开一个小时  \n想法：用户需要预定一个今天晚上 8 点开始，持续约 1 小时的会议。我可以使用 get_info 工具来提取会议信息。  \n工具：get_info  \n工具输入：今天晚上 8 点，开会时长 1 小时  \n'

 **** 本次调用工具： 'get_info'  
 **** 本次输入： '今天晚上 8 点，开会时长 1 小时'
 用户输入：帮我定今天晚上 8 点的会议，大概开一个小时  
想法：用户需要预定一个今天晚上 8 点开始，持续约 1 小时的会议。我可以使用 get_info 工具来提取会议信息。  
工具：get_info  
工具输入：今天晚上 8 点，开会时长 1 小时  
已更新：会议开始时间:2023-08-21 20:00:00
已更新：会议结束时间:2023-08-21 21:00:00

工具输出会议信息更新好了，请提示用户输入如下信息:['会议地点']
' 我需要给用户答复  \n最后输出：会议信息更新好了，但是还需要\'会议地点\'"'
 我需要给用户答复  
最后输出：会议信息更新好了，但是还需要'会议地点'"

> Finished chain.
'会议信息更新好了，但是还需要\'会议地点\'"'
```
```python
agent_executor.run('开会地点定在202')
```
```
> Entering new AgentExecutor chain...
' 用户输入：开会地点定在 202  \n想法：用户提供了会议预定信息。  \n工具：get_info  \n工具输入：今天晚上 8 点，开会时长 1 小时，会议地点：202  \n'

 **** 本次调用工具： 'get_info'  
 **** 本次输入： '今天晚上 8 点，开会时长 1 小时，会议地点：202'
 用户输入：开会地点定在 202  
想法：用户提供了会议预定信息。  
工具：get_info  
工具输入：今天晚上 8 点，开会时长 1 小时，会议地点：202  
已更新：会议地点:202

工具输出会议信息更新好了，请询问用户还有补充的信息吗，如:['会议名称', '召集人名称', '参会人数', '参会人员', '参会单位', '报名截止时间']
' 我需要给用户答复。  \n最后输出：会议信息更新好了，但是还需要\'会议名称\', \'召集人名称\', \'参会人数\', \'参会人员\', \'参会单位\', \'报名截止时间\'这些信息。"'
 我需要给用户答复。  
最后输出：会议信息更新好了，但是还需要'会议名称', '召集人名称', '参会人数', '参会人员', '参会单位', '报名截止时间'这些信息。"

> Finished chain.

会议信息更新好了，但是还需要'会议名称', '召集人名称', '参会人数', '参会人员', '参会单位', '报名截止时间'这些信息。"
```
```python
agent_executor.run("参会人有小李小王")
```
```
> Entering new AgentExecutor chain...
' 用户输入：参会人有小李小王  \n想法：用户提供了会议预定信息。  \n工具：get_info  \n工具输入：参会人：小李小王  \n'

 **** 本次调用工具： 'get_info'  
 **** 本次输入： '参会人：小李小王'
 用户输入：参会人有小李小王  
想法：用户提供了会议预定信息。  
工具：get_info  
工具输入：参会人：小李小王  
已更新：参会人员:['小李', '小王']

工具输出会议信息更新好了，请询问用户还有补充的信息吗，如:['会议名称', '召集人名称', '参会人数', '参会单位', '报名截止时间']
' 我需要询问用户是否还有其他信息需要补充。  \n最后输出：会议信息更新好了，请询问用户还有补充的信息吗，如:[\'会议名称\', \'召集人名称\', \'参会人数\', \'参会单位\', \'报名截止时间\']"'
 我需要询问用户是否还有其他信息需要补充。  
最后输出：会议信息更新好了，请询问用户还有补充的信息吗，如:['会议名称', '召集人名称', '参会人数', '参会单位', '报名截止时间']"

> Finished chain.

'会议信息更新好了，请询问用户还有补充的信息吗，如:[\'会议名称\', \'召集人名称\', \'参会人数\', \'参会单位\', \'报名截止时间\']"'
```

```python
agent_executor.run("明天会议主要是讨论agent的问题")
```

```
> Entering new AgentExecutor chain...
' 用户输入：明天会议主要是讨论 agent 的问题  \n想法：用户提供了会议预定信息。  \n工具：get_info  \n工具输入：明天会议主要是讨论 agent 的问题  \n'

 **** 本次调用工具： 'get_info'  
 **** 本次输入： '明天会议主要是讨论 agent 的问题'
 用户输入：明天会议主要是讨论 agent 的问题  
想法：用户提供了会议预定信息。  
工具：get_info  
工具输入：明天会议主要是讨论 agent 的问题  
已更新：会议名称:讨论 agent 的问题

工具输出会议信息更新好了，请询问用户还有补充的信息吗，如:['召集人名称', '参会人数', '参会单位', '报名截止时间']
' 我需要询问用户是否还有其他需要补充的信息。  \n最后输出：好的，明天会议主要是讨论 agent 的问题。还有其他需要补充的信息吗，如:[\'召集人名称\', \'参会人数\', \'参会单位\', \'报名截止时间\']"'
 我需要询问用户是否还有其他需要补充的信息。  
最后输出：好的，明天会议主要是讨论 agent 的问题。还有其他需要补充的信息吗，如:['召集人名称', '参会人数', '参会单位', '报名截止时间']"

> Finished chain.
'好的，明天会议主要是讨论 agent 的问题。还有其他需要补充的信息吗，如:[\'召集人名称\', \'参会人数\', \'参会单位\', \'报名截止时间\']"'
```

```python
agent_executor.run("没啥问题了")
```

```
> Entering new AgentExecutor chain...
' 用户输入：没啥问题了  \n想法：用户没有提供会议预定信息，我需要使用 create 工具来创建会议。  \n工具：create  \n工具输入：无  \n'

 **** 本次调用工具： 'create'  
 **** 本次输入： '无'
 用户输入：没啥问题了  
想法：用户没有提供会议预定信息，我需要使用 create 工具来创建会议。  
工具：create  
工具输入：无  

工具输出会议已经创建，信息如下：{'会议开始时间': '2023-08-21 20:00:00', '会议结束时间': '2023-08-21 21:00:00', '会议地点': 202, '参会人员': ['小李', '小王'], '会议名称': '讨论 agent 的问题'}
' 我需要给用户答复  \n最后输出：会议已经创建，信息如下：{\'会议开始时间\': \'2023-08-21 20:00:00\', \'会议结束时间\': \'2023-08-21 21:00:00\', \'会议地点\': 202, \'参会人员\': [\'小李\', \'小王\'], \'会议名称\': \'讨论 agent 的问题\'}。如果您还有其他问题，请随时告诉我。"'
 我需要给用户答复  
最后输出：会议已经创建，信息如下：{'会议开始时间': '2023-08-21 20:00:00', '会议结束时间': '2023-08-21 21:00:00', '会议地点': 202, '参会人员': ['小李', '小王'], '会议名称': '讨论 agent 的问题'}。如果您还有其他问题，请随时告诉我。"

> Finished chain.
会议已经创建，信息如下：{'会议开始时间': '2023-08-21 20:00:00', '会议结束时间': '2023-08-21 21:00:00', '会议地点': 202, '参会人员': ['小李', '小王'], '会议名称': '讨论 agent 的问题'}。如果您还有其他问题，请随时告诉我。"
```

### 自己实现代理:增加了根据时间选择地点，弹出人员列表，简化agent步骤
```python
from chatglm import ChatGLM

llm=ChatGLM(temperature=0.1)
#初始化储存字典
def initialize():
    all = {};time = {};room = {};nam = {"参会人员":[]};other = {}
    time_keys = "会议开始时间,会议结束时间";time_keys = time_keys.split(",")
    room_keys = "会议地点";room_keys = room_keys.split(",")
    nam_keys = "参会人员";nam_keys = nam_keys.split(",")
    other_keys = "会议名称,召集人名称,参会单位,报名截止时间";other_keys = other_keys.split(",")
    for key in other_keys:
        other[key] = "default"
    return all, time, time_keys, room, room_keys, nam, nam_keys, other, other_keys
```
会议信息提取工具
```python
from langchain import PromptTemplate
import time

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
get_info_llm = ChatGLM(temperature=0.1)


def get_infos(input: str):
    global get_info_llm
    # prompt template
    promptsingle = PromptTemplate(
        template="""
    你是一款专门提取会议信息的智能助手。你的任务是尽可能在用户输入的一段话中找出与会议相关的信息，其标签必须为以下之一，不要超出范围：
    ["会议开始时间","会议结束时间","会议名称","召集人名称","参会人数","参会人员","参会单位","报名截止时间","会议地点"]。
    
    你要遵循以下规则：
    1.最重要的是：如果用户没有提及的信息就不要更新，你不能编造或添加任何不在用户输入中的信息；
    2.你要把这些信息整理为字典格式并输出，字典中每一项的键是会议信息的标签，值是从用户输入中提取出来的相应信息：{{"标签":"相应信息"}}；
    3.对于时间信息，你需要将输入中的时间更改成如下格式：YYYY-MM-DD HH:MM:SS后再更新信息，如果是"会议开始时间"，你需要结合当前时间{current_time}；如果是"会议结束时间"，你需要参考"会议开始时间"；
    4.对于参会人员信息，你需要将人分开，并储存为列表；
    5.对于参会单位信息，你需要将单位分开，并储存为列表；
    6.对于会议地点信息，如果用户提及了需要创建会议的地点信息，则你要将其转换成阿拉伯数字储存。
    如：
    用户输入：会议定在二零三室
    你的输出：{{"会议地点":203"}}

    在输出结果时，必须严格按照上述规则，请开始你的工作：

    用户输入：{input}
    你的输出：

    """,
        input_variables=["input", "current_time"],
    )
    prompt_text = promptsingle.format(input=input, current_time=current_time)
    # print(prompt_text)
    output = get_info_llm(prompt_text)
    # 清除历史记录，不需要记录
    get_info_llm.history.clear()
    return output

def get_info(input):
    import json

    global rec, rec_keys, other, other_keys, all, rooms, dics ,nam,nam_keys
    str_dics = get_infos(input)
    dics = json.loads(json.loads(str_dics))
    str = "最终输出："
    for key in dics:
        # 如果键在字典中，更新对应的值
        if key in time_keys and dics[key] != None:
            time[key] = dics[key]
            time_keys.remove(key)
            all.update({key: dics[key]})
            str += f"""已更新：{key}:{dics[key]}，"""

        # 必须要先输入时间，并且要选取在rooms中的房间，不然不更新
        if key in room_keys and dics[key] != None and time_keys == []:
            if dics[key] in rooms:
                room[key] = dics[key]
                room_keys.remove(key)
                all.update({key: dics[key]})
                str += f"已更新：{key}:{dics[key]}，"
            else:
                str += "地点信息不对，请重新输入，"
                
        if key == '参会人员' and dics[key] != None:
            for name_i in dics[key]:
                if name_i in nam_keys:
                    nam[key] += name_i
                    str += f"已更新：{key}:{name_i}，"
                else:
                    str += f"人名{name_i}信息不对，请重新输入，"
            all.update({key: dics[key]})

        if key in other_keys and dics[key] != "":
            other[key] = dics[key]
            other_keys.remove(key)
            all.update({key: dics[key]})
            str += f"已更新：{key}:{dics[key]}，"

    if time_keys != []:
        out = f"{str}请输入如下信息:{[key for key in time_keys]}，"
        # print(f)
        return out
    elif room_keys != []:
        # 只有当时间信息处理好后才更新会议室信息
        rooms_dic = {"203": "10人", "204": "20人", "205": "30人"}  # 此处替换成会议室获取函数
        rooms = [key for key in rooms_dic]  # 可用会议室列表
        out = f"""{str}根据您输入的会议时间，当前可用的会议室有：{rooms_dic}"""+'\n'
        return out
    elif nam_keys != [] and nam["参会人员"]==[]:
        # 当会议室获取好后会弹出会议人姓名
        out = f"{str}您可以选择参会人:{nam_keys}"
        return out
    elif other_keys != []:
        out = f"{str}请问您还有补充的信息吗，如:{[key for key in other_keys]}"
        return out
    else:
        out = f"{str}，会议信息更新好了"
        return out
```
创建会议工具
```python
def create(a):
    global all
    str = f"会议已经创建，信息如下：{all}"
    return str

```
创建代理提示
```python
template = """
你是一个预定会议的代理，你的任务是判断用户是否要预定会议，并确定要使用的工具或者不使用。你会接受自然语言输入，并转换为合理的输出。
你必须根据你所处的对话情况做出工具选择，有三个情况，请依次判断，不要乱编：
1:会议信息预定：当用户想预定会议，并且提供了预定会议信息，则使用"get_info_tool"；
2:结束：当用户不想提供信息时，会议预定结束，则使用"create_tool"；
3：闲聊：当用户的输入不包含会议预定信息，并且不是情况2时，用户是在闲聊，则使用"none"。
你的输出必须严格遵守以下格式：
  想法：用户的意图是干什么 \n工具：要使用的工具 \n工具输入：按照本轮的用户输入来选取，不要有所省略\n

示例：
历史对话：
'帮我定明天上午六点的会议；工具：get_info_tool，\n工具输入：明天上午六点的会议\n"'
'大概要开三小时；工具：get_info_tool，\n工具输入：明天上午六点的会议，大概要开三小时\n"'

用户输入：定203会议室
你的回答:
想法：用户想预定会议工具：get_info_tool\\n 工具：get_info_tool \\n工具输入：定 203 会议室"

开始吧！
历史对话：\n{history}
用户输入：\n{input}
你的回答:
"""

from langchain import PromptTemplate, LLMChain

prompt = PromptTemplate(template=template, input_variables=["history","input"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
```
代理输出解析器，解析为工具和工具输入
```python
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

def parse(llm_output):
    llm_output = llm_output.replace("\\n", "\n")
    llm_output = llm_output.replace('\\', '')

    print(repr(llm_output))
    regex = r"工具\s*\d*\s*[:：]\s*(.*?)\s*\n?\s*(?:工具输入)\s*\d*\s*[:：][\s]*(.*)\n?"
    match = re.search(regex, llm_output, re.DOTALL)

    if not match:#没匹配到
        print("出错了")
        return 'none','none'
    
    action = match.group(1).strip()
    action_input = match.group(2).strip()
    print("\n **** 本次调用工具：", repr(action),"\n **** 本次工具输入：", repr(action_input))
    return action,action_input
```
根据的解析调用工具
```python
def to_tools(input,parsed_output):
    '''
    此函数为根据解析的agent输出来调用工具并获得输出
    '''
    action,action_input=parsed_output
    if 'none' in action:
        return llm(input)
    if 'get_info_tool' in action:
        return get_info(action_input)
    if 'create_tool' in action:
        return create(' ')
    else:
        return '出错了'
```

测试
```python
#初始化
all, time, time_keys, room, room_keys, nam, nam_keys, other, other_keys = initialize()
rooms = [];nam_keys= ["小李", "小王", "小明","小张"]# 此处替换成会议人获取函数
```

```python
history=''
input='帮我定明天上午六点的会议'

agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```

```
**** 本次调用工具： 'get_info_tool' 
 **** 本次工具输入： '明天上午六点的会议"'
最终输出：已更新：会议开始时间:2023-08-25 06:00:00，请输入如下信息:['会议结束时间']，
```

```python
#更新历史消息
history+='用户输入：'+input+'\n'
history+='你的回答：'+f'工具：{parsed_output[0]}，工具输入：{parsed_output[1]}'+'\n'
print(history)
```
```
用户输入：帮我定明天上午六点的会议
你的回答：工具：get_info_tool，工具输入：明天上午六点的会议"
```
```python
input='大概开三小时'
agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```
```
**** 本次调用工具： 'get_info_tool，' 
 **** 本次工具输入： '明天上午六点的会议，大概开三小时"'
最终输出：已更新：会议结束时间:2023-08-25 09:00:00，根据您输入的会议时间，当前可用的会议室有：{'203': '10人', '204': '20人', '205': '30人'}
```

```python
history+='用户输入：'+input+'\n'
history+='你的回答：'+f'工具：{parsed_output[0]}，工具输入：{parsed_output[1]}'+'\n'
print(history)
```
```
用户输入：帮我定明天上午六点的会议
你的回答：工具：get_info_tool，工具输入：明天上午六点的会议"
用户输入：大概开三小时
你的回答：工具：get_info_tool，，工具输入：明天上午六点的会议，大概开三小时"
```

```python
input='去205开会吧'

agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```
```
**** 本次调用工具： 'get_info_tool' 
 **** 本次工具输入： '去 205 开会吧"'
最终输出：已更新：会议地点:205，您可以选择参会人:['小李', '小王', '小明', '小张']
```
```python
history+='用户输入：'+input+'\n'
history+='你的回答：'+f'工具：{parsed_output[0]}，工具输入：{parsed_output[1]}'+'\n'
print(history)
```
```
用户输入：帮我定明天上午六点的会议
你的回答：工具：get_info_tool，工具输入：明天上午六点的会议"
用户输入：大概开三小时
你的回答：工具：get_info_tool，，工具输入：明天上午六点的会议，大概开三小时"
用户输入：去205开会吧
你的回答：工具：get_info_tool，工具输入：去 205 开会吧"
```
```python
input='小王小张要参加'

agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```
```
**** 本次调用工具： 'get_info_tool，' 
 **** 本次工具输入： '小王小张要参加""'
最终输出：已更新：参会人员:小王，已更新：参会人员:小张，请问您还有补充的信息吗，如:['会议名称', '召集人名称', '参会单位', '报名截止时间']
```
```python
history+='用户输入：'+input+'\n'
history+='你的回答：'+f'工具：{parsed_output[0]}，工具输入：{parsed_output[1]}'+'\n'
```

```python
input='明天主要讨论知识分享'

agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```
```
**** 本次调用工具： 'get_info_tool，' 
 **** 本次工具输入： '明天主要讨论知识分享"'
最终输出：已更新：会议名称:知识分享，请问您还有补充的信息吗，如:['召集人名称', '参会单位', '报名截止时间']
```
```python
history+='用户输入：'+input+'\n'
history+='你的回答：'+f'工具：{parsed_output[0]}，工具输入：{parsed_output[1]}'+'\n'
```
```python
input='没有了'

agent_output=llm_chain.run(input=input,history=history)
parsed_output=parse(agent_output)
final=to_tools(input,parsed_output)
print(final)
```
```
 **** 本次调用工具： 'create_tool' 
 **** 本次工具输入： '没有了"'
会议已经创建，信息如下：{'会议开始时间': '2023-08-25 06:00:00', '会议结束时间': '2023-08-25 09:00:00', '会议地点': '205', '参会人员': ['小王', '小张'], '会议名称': '知识分享'}
```