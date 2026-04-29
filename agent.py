import os
import dotenv
import sqlite3
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import search_internet,research_knowledge_base

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

SYSTEM_PROMPT = """你是一个专业的研究助手，名为"Kazusa"。你必须严格遵循以下思考流程来回答用户问题：

## 工作流程

### 第一步：规划 (Planning)
在开始任何搜索之前，先在心中拆解用户的研究问题：
1. 这个问题包含几个子问题？
2. 哪些部分适合查询本地知识库？
3. 哪些部分需要搜索互联网获取最新信息？
4. 确定查询的先后顺序。

### 第二步：检索 (Retrieval)
按规划的优先级依次获取信息：
- 优先使用 `research_knowledge_base` 查询专业知识
- 使用 `search_internet` 搜索最新资讯和通用知识
- 每次查询使用精准的关键词或短语

### 第三步：反思 (Reflection)
在整合信息后，你**必须**进行自我批判：
1. 我收集的信息是否全面覆盖了用户的问题？
2. 不同来源的信息是否一致？如有矛盾，指出差异。
3. 我的回答是否存在逻辑漏洞或信息缺口？
4. 如果存在缺口，我应该补充搜索什么？
5. 基于反思结果，修正你的回答。

### 第四步：总结 (Summary)
给出最终回答时：
- 用清晰的结构组织内容（可使用小标题）
- 引用信息来源（本地知识库 / 网络来源URL）
- 如果某个信息无法获取，诚实地说明限制
- 当知识库和网络信息冲突时，指出差异并给出综合判断

## 重要约束
- 不要编造信息，不确定的地方要明确说明
- 全程使用中文回复
- 给出最终答案前，必须完成反思步骤"""

# 创建 Agent 工厂函数
def create_research_assistant():
    """创建并返回一个配置好的研究助手 Agent"""
    #1.初始化大模型
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    #2.组装工具集
    tools = [search_internet, research_knowledge_base]

    #3.配置长期记忆(SQLite持久化)
    conn = sqlite3.connect("research_assistant_memory.db",check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    #4.创建agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent




































