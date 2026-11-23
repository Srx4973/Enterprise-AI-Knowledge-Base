"""
【FastAPI 启动入口】
负责初始化应用、配置跨域中间件、挂载路由。
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.api.routes import router

# 初始化 FastAPI
app = FastAPI(
    title="Enterprise AI Knowledge Base API",
    description="基于 RAG 与 Agent 的企业级知识库后端",
    version="1.0.0"
)

# 配置 CORS（跨域资源共享）
# 因为前端可能运行在 8080 端口，后端在 8000 端口，不加这个前端会报错
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议改成前端的具体地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由（统一加 /api 前缀）
app.include_router(router, prefix="/api")

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok", "message": "Backend is running!"}

@app.get("/")
def index():
    file_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    return FileResponse(file_path)