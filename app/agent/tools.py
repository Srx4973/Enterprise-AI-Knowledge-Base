from chromadb import PersistentClient
from ..core.embedder import embedding_fn
from ..rag.retriever import build_context_and_prompt
from ..core.config import CHROMA_PATH

# ---------- 工具定义 ----------
def get_weather(city: str) -> str:
    if "太原" in city: return "太原今天晴，气温 28 度，微风。"
    return f"{city}今天多云，气温 25 度。"


def calculate(expression: str) -> str:
    try:
        return f"计算结果是：{eval(expression)}"
    except Exception as e:
        return f"计算失败：{e}"


#独立初始化数据库连接，避免循环导入
_tool_client = PersistentClient(path=CHROMA_PATH)
_tool_collection = _tool_client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_fn
)

def list_all_documents() -> str:
    """获取知识库中所有的文档列表"""
    # include=[] 表示只要元数据，不需要把几千个文本块全读出来，极快
    all_data = _tool_collection.get(include=["metadatas"])

    if not all_data or not all_data.get("metadatas"):
        return "知识库目前是空的。"

    # 用 set 去重，提取所有的文件名
    sources = list(set([meta.get("source", "未知") for meta in all_data["metadatas"] if meta]))
    if not sources:
        return "知识库中暂未找到有效文档。"

    return f"当前知识库共有 {len(sources)} 个文档，分别是：{', '.join(sources)}"

#rag
def search_documents(query: str) -> str:
    # 1. 获取提示词和参考来源
    prompt, source_info = build_context_and_prompt(query)
    if not prompt:
        return "根据当前文档无法回答。"
    # 2.把原本查出来的资料（prompt），加上来源信息，直接打包返回
    sources = "\n\n".join(source_info)
    # 3.直接返回原汁原味的资料给外面的 Agent 大模型 A
    return f"""[参考资料开始]
    {prompt}
    [参考资料结束]

    系统强制指令：请严格基于上述参考资料回答，不需要输出参考来源。
    [SOURCES_START]
    {sources}
    [SOURCES_END]
    """


agent_tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询指定城市的天气情况",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算数学表达式",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                                                     "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_documents", "description": "检索本地知识库获取专业信息。注意：如果用户同时询问了多个独立的问题，请分多次调用此工具，每次只查一个具体问题。",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "list_all_documents","description": "当用户询问知识库里一共有几个文档、有哪些文件、库里有什么资料时，调用此工具",
                                      "parameters": {"type": "object", "properties": {}}
    }}
]