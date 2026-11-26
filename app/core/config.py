"""
【全局配置模块】
负责加载环境变量、初始化 Celery 实例，以及维护全局常量。
"""
import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()
#向量库
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "../chroma_db_online")
CHROMA_COLLECTION = "pdf_docs"

#api配置
# 模型配置（小任务）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = "GLM-4-Flash-250414"
DEEPSEEK_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
#最终模型(生成)配置
END_MODEL = 'zai-org/GLM-5.3-Flash'
END_URL = 'https://api.siliconflow.cn/v1/chat/completions'

# 向量模型（检索）的配置
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_URL  = "https://api.siliconflow.cn/v1/embeddings"
#重排配置
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"

#分块参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_SIZE = 10
TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}

#分步任务
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

# 配置 Redis 用于记忆存储
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB_MEMORY = 2  # 用 2 号库专门存聊天记录