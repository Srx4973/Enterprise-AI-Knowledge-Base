import requests
from chromadb import PersistentClient
from ..core.embedder import embedding_fn
from ..core.config import EMBEDDING_API_KEY,RERANK_MODEL,RERANK_URL,CHROMA_PATH

# 1. 连接数据库
client = PersistentClient(path=CHROMA_PATH)

# 2. 获取集合
collection = client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


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

    # 拼上下文和来源
    context_parts, source_info = [], []
    for i, (doc, meta) in enumerate(zip(final_docs, final_metas)):
        location = meta.get('location', '未知')
        context_parts.append(f"[片段{i + 1}] {doc}")
        source_info.append(f"片段{i + 1} 来自第 {location} 页")

    context = "\n\n".join(context_parts)
    sources = "\n".join(source_info)

    prompt = f"""请基于以下上下文回答问题。如果上下文不包含相关信息，请说“根据当前文档无法回答”。

上下文：
{context}

问题：{question}
回答："""
    return prompt, sources