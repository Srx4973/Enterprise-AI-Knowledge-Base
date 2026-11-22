"""
【异步任务模块】
负责接收 Celery 任务指令，调度文件解析、分块、清洗、向量化并存入 ChromaDB。
"""
import os
from chromadb import PersistentClient

from ..core.config import celery_app, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from ..core.embedder import embedding_fn
from ..preprocess.parsers import load_pdf_pages, load_docx, load_txt, load_csv, load_excel
from ..preprocess.chunker import build_chunks


@celery_app.task(bind=True)
def process_document_to_db(self, file_path: str):
    """解析 → 分块 → 过滤 → 入库。分块统一走 build_chunks。"""
    # 更新初始状态
    self.update_state(state="PROGRESS", meta={"progress": 5, "message": "正在读取 PDF 页面..."})

    ext = os.path.splitext(file_path)[1].lower()

    # ---------- 1. 解析文件 ----------
    try:
        if ext == ".pdf":
            pages = load_pdf_pages(file_path)
        elif ext == ".docx":
            pages = load_docx(file_path)
        elif ext == ".txt":
            pages = load_txt(file_path)
        elif ext == ".csv":
            pages = load_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            pages = load_excel(file_path)
        else:
            return {"status": "error", "message": f"不支持的文件格式: {ext}"}
    except Exception as e:
        return {"status": "error", "message": f"文件读取失败: {e}"}

    self.update_state(state="PROGRESS", meta={"progress": 15, "message": "正在分块与清洗数据..."})

    # ---------- 2. 分块（关键：走 build_chunks，overlap 只传一次） ----------
    # 文本块自动重叠 CHUNK_OVERLAP；表格块内部自动忽略 overlap
    all_chunks = build_chunks(
        pages=pages,
        ext=ext,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    # ---------- 3. 过滤 ----------
    filtered: list[dict] = []
    for c in all_chunks:
        text = c["text"].strip()
        if len(text) < 20:                       # 太短的块无检索价值
            continue
        dot_count = text.count(".") + text.count("…")
        if  dot_count > len(text)*0.2:   # 标点密度过高，疑似目录/乱码
            continue
        filtered.append(c)

    if not filtered:
        return {"status": "error", "message": "未提取到有效文本块"}

    # ---------- 4. 连库 ----------
    self.update_state(state="PROGRESS", meta={"progress": 20, "message": "连接数据库..."})
    client = PersistentClient(path=CHROMA_PATH)

    source = os.path.basename(file_path)

    # # 单文件场景：每次重建 collection，保证幂等
    # try:
    #     client.delete_collection("pdf_docs")
    # except Exception:
    #     pass

    collection = client.get_or_create_collection(
        name="pdf_docs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    #多文件场景：先删本文件旧块，再插新块（幂等重入）
    try:
        old_ids = collection.get(where={"source": source})["ids"]
        if old_ids:
            collection.delete(ids=old_ids)
    except Exception:
        pass

    # ---------- 5. 批量入库 ----------
    batch_size = 100
    total_steps = (len(filtered) + batch_size - 1) // batch_size

    for step, i in enumerate(range(0, len(filtered), batch_size), start=1):
        batch = filtered[i: i + batch_size]

        # ids 用 "source::chunk_id"：多文件不撞 id，且和 chunk_id 一一对应
        ids = [f"{source}::{c['chunk_id']}" for c in batch]

        # metadata 带上 chunk_id / segment：检索回来能区分同一 location 的多块
        metadatas = [
            {
                "source": source,
                "location": c["location"],    # 给用户 / prompt 看
                "chunk_id": c["chunk_id"],    # 系统主键
                "segment": c["segment"],      # 排序用
            }
            for c in batch
        ]

        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=metadatas,
            ids=ids,
        )

        progress = int(20 + (step / total_steps) * 70)
        self.update_state(
            state="PROGRESS",
            meta={
                "progress": progress,
                "message": f"正在向量化并入库: 已处理 {step}/{total_steps} 个批次...",
            },
        )

    # ---------- 6. 完成 ----------
    return {
        "status": "success",
        "message": f"成功插入 {len(filtered)} 个文本块",
        "file": source,          # 原来写的是 pdf_path，修正为 file_path 的 basename
    }