import requests
from chromadb import PersistentClient
from ..core.memory import get_history
from ..core.embedder import embedding_fn
from ..core.config import EMBEDDING_API_KEY,RERANK_MODEL,RERANK_URL,CHROMA_PATH,DEEPSEEK_API_KEY,MODEL_NAME,DEEPSEEK_URL

# 1. 连接数据库
client = PersistentClient(path=CHROMA_PATH)

# 2. 获取集合
collection = client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

#通过上下文精确用户问题
def rewrite_query(query: str, session_id: str) -> str:
    """利用大模型结合历史记录，将追问改写为独立完整的查询句"""
    history = get_history(session_id)
    # 如果历史记录很短，说明是第一句，不用改写，直接返回
    if len(history) < 2:
        return query

    # 构建改写提示词
    prompt = f"""你是一个智能搜索助手。请结合历史对话记录，将用户的最新问题改写为一个完整、独立、适合去搜索引擎检索的句子。
只需要输出改写后的句子，不要输出任何解释。

历史对话：
{history[-4:]}  # 只取最近两轮，省 Token

用户最新问题：{query}
改写后的检索词："""

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        resp = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=data)
        if resp.status_code == 200:
            new_query = resp.json()["choices"][0]["message"]["content"].strip()
            print(f" 查询改写: [{query}] -> [{new_query}]")
            return new_query
    except Exception as e:
        print(f"改写失败，使用原问题: {e}")

    return query

# ---------- 检索函数 ----------
def retrieve(query, n_results):
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]
    return docs, metas, distances

#---------- 重排函数 ----------
def rerank_documents(query, docs, top_n=3):
    """调用 API 对候选文档进行重排"""
    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": docs,
        "top_n": top_n
    }
    resp = requests.post(RERANK_URL, headers=headers, json=data)

    if resp.status_code != 200:
        print(f"Rerank API 报错: {resp.text}")
        # 如果重排挂了，退化为直接返回前 top_n 个原始文档的索引
        return list(range(min(top_n, len(docs))))

    # 解析返回结果，提取排好序的原始索引
    results = resp.json()["results"]
    return [item["index"] for item in results]

# -----检索、过滤、重排，拼 Prompt-----
def build_context_and_prompt(question: str):
    # 1. 粗排：去库里拿 10 个候选块
    docs, metas, distances = retrieve(question, n_results=10)

    # 2. 过滤低相似度块
    valid = [(d, m, dist) for d, m, dist in zip(docs, metas, distances) if dist < 1.5]
    if not valid:
        return None, None

    docs, metas, _ = zip(*valid)
    docs, metas = list(docs), list(metas)

    # 3. 精排：把这 10 个块丢给 Rerank API，挑出得分最高的 3 个
    print(f"粗排拿到 {len(docs)} 个候选块，正在重排...")
    top_indices = rerank_documents(question, docs, top_n=3)

    # 4. 根据重排返回的索引，提取最终的文本和元数据
    final_docs = [docs[i] for i in top_indices]
    final_metas = [metas[i] for i in top_indices]

    print(f"重排完毕，开始拼接上下文...")
    # 拼上下文和来源
    context_parts, source_info = [], []
    for i, (doc, meta) in enumerate(zip(final_docs, final_metas)):
        # 取出文件名和页码
        source_name = meta.get('source', '未知文件')
        location = meta.get('location', '未知')

        # 【给大模型看的上下文】保留片段标号，方便它区分不同段落
        context_parts.append(f"[片段{i + 1}] {doc}")

        # 【给用户看的参考来源】改成：《文件名》 第x页
        source_info.append(f"《{source_name}》 {location}")

    #如果检索出来的3个片段都来自同一本书的同一页，可以去重展示
    # dict.fromkeys 可以在去重的同时，保持原有的顺序不乱
    source_info = list(dict.fromkeys(source_info))

    context = "\n\n".join(context_parts)


    prompt = f"""请基于以下参考信息回答问题。
如果参考信息中直接包含了答案，请详细作答。
如果参考信息中没有直接提及，但包含相关的技术特性或背景描述，请尽量总结现有信息来回答用户。
只有在参考信息与用户问题【完全毫无关联】时，才请回答：“根据当前文档无法回答”。

参考信息：
{context}

用户问题：{question}
回答："""
    return prompt, source_info