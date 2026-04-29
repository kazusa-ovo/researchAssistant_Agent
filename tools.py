import os
import dotenv
import requests
from tenacity import retry,stop_after_attempt,wait_exponential,retry_if_exception_type
from requests.exceptions import RequestException, Timeout, ConnectionError
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

dotenv.load_dotenv()
# 网络瞬时错误：超时和连接中断
RETRYABLE_NETWORK_ERRORS = (Timeout, ConnectionError)
# 一个可复用的降级消息
FALLBACK_MESSAGE = "该服务暂时不可用，请稍后重试或基于已有知识回答。"

# 底层带重试的网络请求函数
@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RETRYABLE_NETWORK_ERRORS),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _perform_web_search(query:str) -> list:
    """执行网络搜索，仅重试网络层面的瞬时错误"""
    # 检查API_KEY
    if not os.environ["TAVILY_API_KEY"]:
        raise ValueError("TAVILY_API_KEY 环境变量未设置，无法进行网络搜索TvT")
    search_tool = TavilySearchResults(
        api_key=os.environ["TAVILY_API_KEY"],
        max_results=5,
    )
    return search_tool.invoke({"query": query})

# Agent 可用的工具函数
@tool
def search_internet(query:str) -> list:
    """使用此工具搜索互联网，获取最新信息、新闻或任何需要实时数据的问题。查询参数应是一个或多个准确的关键词。"""
    print(f"正在搜索:{query}")
    try:
        results = _perform_web_search(query)
        if not results:
            return "网络搜索未返回结果TvT"
        formatted = []
        for res in results:
            content = res.get("content","")
            url = res.get("url","")
            formatted.append(f"· {content}" + (f" (来源: {url})" if url else ""))
        return "以下是从网络上搜索到的相关信息：\n" + "\n".join(formatted)
    except ValueError as ve:
        return f"网络搜索工具配置错误：{ve}"
    except Exception as e:
        print(f"网络搜索失败，触发降级。错误: {e}")
        return f"{FALLBACK_MESSAGE}"

# 底层带重试的文档检索函数
@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _retrieve_docs(query:str,retriever):
    """执行文档检索，带重试机制"""
    return retriever.invoke(query)

# 初始化 RAG 检索器 (带降级)
def initialize_rag_retriever(file_path="knowledge_base.txt"):
    """初始化 RAG 知识库，如果失败则返回 None (降级)"""
    try:
        loader = TextLoader(file_path,encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150
        )
        documents = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever()
        print("RAG知识库加载成功")
        return retriever
    except Exception as e:
        print(f"RAG 知识库初始化失败 (可能缺少 'knowledge_base.txt' 文件): {e}")
        print("已启动降级策略：研究助手将仅依赖网络搜索和基础模型能力。")
        return None

# 初始化检索器 (在文件加载时执行)
_brain_retriever = initialize_rag_retriever()

# Agent课调用的工具函数
@tool
def research_knowledge_base(query:str) -> str:
    """在本地知识库中搜索相关专业知识，用于查找关于技术、公司政策、内部文档等特定信息。查询应是一个关键短语或问题。"""
    if _brain_retriever is None:
        print("本地知识库不可用，请尝试使用网络搜索工具。")
    print(f"正在查询本地知识库:{query}")
    try:
        relevant_docs = _brain_retriever.invoke(query)
        if not relevant_docs:
            return "知识库中没有找到相关信息"
        formatted = []
        for doc in relevant_docs:
            formatted.append(f"· {doc.page_content}")
        return "以下是从本地知识库中找到的相关信息：\n" + "\n".join(formatted)
    except Exception as e:
        print(f"知识库查询失败，触发降级。错误: {e}")
        return f"{FALLBACK_MESSAGE}"

























