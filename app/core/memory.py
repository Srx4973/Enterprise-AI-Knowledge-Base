import json
import redis
from .config import REDIS_HOST, REDIS_PORT, REDIS_DB_MEMORY

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_MEMORY, decode_responses=True)

MAX_HISTORY_TURNS = 5  # 只保留最近 5 轮对话，防止 Token 爆炸


def get_history(session_id: str) -> list:
    """从 Redis 获取历史记录"""
    data = redis_client.get(f"chat_history:{session_id}")
    if data:
        return json.loads(data)
    return []


def save_history(session_id: str, messages: list):
    """保存历史记录，并做滑动窗口截断"""
    # 如果超过最大轮数，截取最后 MAX_HISTORY_TURNS * 2 条（因为一问一答是2条）
    if len(messages) > MAX_HISTORY_TURNS * 2:
        messages = messages[-MAX_HISTORY_TURNS * 2:]

    redis_client.set(f"chat_history:{session_id}", json.dumps(messages), ex=3600)  # 1小时过期


# 获取摘要
def get_summary(session_id: str) -> str:
    return redis_client.get(f"chat_summary:{session_id}") or ""

# 保存摘要（设置过期时间）
def save_summary(session_id: str, summary: str):
    redis_client.set(f"chat_summary:{session_id}", summary, ex=3600)

