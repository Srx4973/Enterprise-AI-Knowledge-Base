"""
【向量化模块】
封装在线 Embedding API 的调用逻辑，为文档和查询提供统一的向量化接口。
"""
import requests
from app.core.config import EMBEDDING_API_KEY, EMBEDDING_URL,EMBEDDING_MODEL

# ---------- 加载embedding模型（线上） ----------
class OnlineEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "online_bge_m3"

    def __call__(self, input):
        # 容错处理：ChromaDB 有时传字符串，有时传列表，统一转成列表
        if isinstance(input, str):
            input = [input]

        # 调用在线 API 获取向量
        headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}
        data = {"model": EMBEDDING_MODEL, "input": input}
        resp = requests.post(EMBEDDING_URL, headers=headers, json=data)

        # 检查状态码（防止 Key 填错或者欠费导致解析失败）
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API 失败: {resp.status_code} {resp.text}")

        # 返回向量列表
        return [item["embedding"] for item in resp.json()["data"]]
    #专门给查询用的方法
    def embed_query(self, input):
        return self.__call__(input)

    # 专门给文档入库用的方法（有些版本需要）
    def embed_documents(self, input):
        return self.__call__(input)

embedding_fn = OnlineEmbeddingFunction()
