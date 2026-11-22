import requests
from chromadb import PersistentClient

from ..core.embedder import embedding_fn
from ..rag.retriever import build_context_and_prompt
from ..core.config import DEEPSEEK_API_KEY,MODEL_NAME,DEEPSEEK_URL,CHROMA_PATH

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
    # include=[] 表示我们只要元数据，不需要把几千个文本块全读出来，极快
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
    prompt, source_info = build_context_and_prompt(query)
    if not prompt: return "根据当前文档无法回答。"

    data = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    resp = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=data)
    if resp.status_code == 200:
        answer = resp.json()["choices"][0]["message"]["content"]
        #  核心优化：把来源信息直接打包给大模型，让它自己复述
        sources = "\n".join(source_info)
        return f"{answer}\n\n参考来源：\n{sources}"
    return "RAG 检索服务失败"


agent_tools = [
    {"type": "function", "function": {"name": "get_weather", "description": "查询指定城市的天气情况",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "计算数学表达式",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                                                     "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_documents", "description": "当用户询问专业知识、大模型、文档内容时，使用此工具检索本地知识库",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "list_all_documents","description": "当用户询问知识库里一共有几个文档、有哪些文件、库里有什么资料时，调用此工具",
                                      "parameters": {"type": "object", "properties": {}}
    }}
]