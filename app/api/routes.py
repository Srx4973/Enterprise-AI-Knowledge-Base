import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

# 导入核心组件
from ..core.config import celery_app, BASE_DIR
from ..tasks.ingest import process_document_to_db
from ..rag.generator import ask_question_stream
from ..agent.agent_main import ask_agent_stream

router = APIRouter()

# 创建存放上传文件的临时目录
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= 1. 文档上传与入库接口 =================
@router.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """接收前端上传的多个文件，保存并触发 Celery"""
    task_ids = []
    for file in files:
        if not file.filename: continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 每个文件派发一个 Celery 任务
        task = process_document_to_db.delay(file_path)
        task_ids.append({"filename": file.filename, "task_id": task.id})

    return {"status": "success", "tasks": task_ids}


# ================= 2. 任务进度查询接口 =================
@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """前端轮询这个接口，获取 Celery 任务的实时进度"""
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == "PENDING":
        return {"state": "PENDING", "progress": 0, "message": "任务正在排队中..."}
    elif task_result.state == "PROGRESS":
        # 返回 update_state 里记录的 meta 数据
        return {
            "state": "PROGRESS",
            "progress": task_result.info.get("progress", 0),
            "message": task_result.info.get("message", "")
        }
    elif task_result.state == "SUCCESS":
        return {"state": "SUCCESS", "result": task_result.result}
    else:
        return {"state": task_result.state, "message": str(task_result.info)}


# ================= 3. 问答请求模型 =================
class AskRequest(BaseModel):
    question: str
    session_id: str = "default_user"

# ================= 4. 纯 RAG 流式问答接口 =================
@router.post("/chat/rag")
async def rag_chat_api(req: AskRequest):
    """不带工具调用的纯 RAG 问答，直接返回流式响应"""
    return StreamingResponse(ask_question_stream(req.question,req.session_id), media_type="text/event-stream")


# ================= 5. Agent 流式问答接口 (带工具调用) =================
@router.post("/chat/agent")
async def agent_chat_api(req: AskRequest):
    """带意图识别和工具调用的 Agent 问答"""
    return StreamingResponse(ask_agent_stream(req.question,req.session_id), media_type="text/event-stream")