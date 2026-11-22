"""
【文本切分模块】
负责将解析后的长文本，智能切分为适合大模型上下文和向量检索的 Chunk。
包含长句硬切兜底、表格表头继承、Overlap 重叠防语义截断等高级策略。
"""
from typing import Iterable
from ..core.config import TABLE_EXTENSIONS, MIN_CHUNK_SIZE

# ==================== 工具函数 ====================

def _hard_split(text: str, size: int) -> list[str]:
    """按字符硬切，绝不丢数据；size 保底 1，防 range 步长为 0 报错"""
    size = max(int(size), 1)
    if not text:
        return []
    return [text[i: i + size] for i in range(0, len(text), size)]


def _emit(chunks: list[dict], location: str, pieces: Iterable[str]) -> None:
    """
    统一出口。三条铁律：
      1. location 始终是业务坐标（用户 / prompt 看它）
      2. chunk_id 永远统一为 "location#N"（无单块特例）
      3. segment 永远为 N（从 1 起），便于排序
    """
    pieces = [p.strip() for p in pieces if p and p.strip()]
    for index, piece in enumerate(pieces, start=1):
        chunks.append({
            "location": location,
            "chunk_id": f"{location}#{index}",
            "segment": index,
            "text": piece,
        })


# ==================== 文本切分 ====================

def split_text(text: str, chunk_size: int, overlap: int=0) -> list[str]:
    """
    三级切分：段落 → 句子 → 硬切。
    保证：① 单块 ≤ chunk_size  ② 长句硬切不丢字  ③ overlap 只与相邻原始块重叠
    """
    chunk_size = max(int(chunk_size), MIN_CHUNK_SIZE)   # 入口保底

    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            pieces.append(current.strip())
        current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current += para + "\n\n"
            continue

        flush()

        if len(para) <= chunk_size:
            current = para + "\n\n"
            continue

        sentences = (
            para.replace("！", "。").replace("？", "。")
                .replace("!", "。").replace("?", "。")
                .split("。")
        )
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_full = sent + "。"

            if len(current) + len(sent_full) <= chunk_size:
                current += sent_full
                continue

            flush()

            if len(sent_full) > chunk_size:
                pieces.extend(_hard_split(sent_full, chunk_size))
                current = ""
            else:
                current = sent_full

    flush()

    # ---------- overlap：只取相邻原始块，不跨块传染；空块直接跳过 ----------
    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    merged: list[str] = [pieces[0]]
    for i in range(1, len(pieces)):
        if not pieces[i].strip():        # 空块不进 merged，也不制造纯重叠块
            continue
        prev_raw = pieces[i - 1]         # 原始块，绝不用 merged[-1]
        ov = min(overlap, len(prev_raw) // 2, chunk_size // 4)
        if ov <= 0 or len(pieces[i]) + ov > chunk_size:
            merged.append(pieces[i])
        else:
            merged.append(prev_raw[-ov:] + pieces[i])

    return merged


# ==================== 主入口 ====================

def build_chunks(
    pages: Iterable[tuple[str, str]],
    ext: str,
    chunk_size: int,
    overlap: int=0,
) -> list[dict]:
    """
    构建 chunks。
      - 结构化数据（CSV/Excel/Word 表格）：按行累加，超长表头截断后继承
      - 非结构化数据（PDF/TXT/Word 段落）：段落→句子→硬切
      - 表格不做 overlap（避免重复行），文本做 overlap
      - location 干净；唯一性由 chunk_id / segment 承担
    """
    chunk_size = max(int(chunk_size), MIN_CHUNK_SIZE)   # 入口保底，下游 budget 永远为正
    normalized_ext = ext.lower()
    chunks: list[dict] = []

    for location, text in pages:
        text = text.strip()
        if not text:
            continue

        # ============ 1. 表格 ============
        is_table = normalized_ext in TABLE_EXTENSIONS or "表格" in location

        if is_table:
            if len(text) <= chunk_size:
                _emit(chunks, location, [text])
                continue

            lines = [ln for ln in text.split("\n") if ln.strip()]

            # 全空白表格：无有效内容，直接跳过（防未来逻辑改动后 lines[0] 越界）
            if not lines:
                continue

            if len(lines) < 2:
                _emit(chunks, location, _hard_split(text, chunk_size))
                continue

            header, data_lines = lines[0], lines[1:]

            # 缺陷 3 修复：来源定大方向，header 结构做兜底
            # header 与 data_lines 同源，"|" 在则必是 Word 表格的 | 分隔格式；
            # 不看 data_lines 的 ":"，避免单元格业务冒号（如 "备注: 无"）导致丢表头
            is_structured = normalized_ext in TABLE_EXTENSIONS
            need_header = (not is_structured) and ("|" in header)

            if need_header and len(header) > chunk_size // 4:
                header = header[: max(chunk_size // 4 - 3, 1)] + "..."

            header_prefix = f"表头: {header}\n" if need_header else ""
            # budget 双保险：chunk_size 已在入口 ≥ 10，这里再 max(..., 1) 防极端
            budget = max(chunk_size - len(header_prefix), chunk_size // 2, 1)

            pieces: list[str] = []
            current = ""
            for line in data_lines:
                if len(current) + len(line) + 1 <= budget:
                    current += line + "\n"
                else:
                    if current.strip():
                        pieces.append(current.strip())
                        current = ""
                    if len(line) + 1 > budget:
                        pieces.extend(_hard_split(line, budget))
                    else:
                        current = line + "\n"
            if current.strip():
                pieces.append(current.strip())

            if need_header:
                pieces = [header_prefix + p for p in pieces]

            _emit(chunks, location, pieces)
            continue

        # ============ 2. 文本 ============
        pieces = split_text(text, chunk_size=chunk_size, overlap=overlap)
        _emit(chunks, location, pieces)

    return chunks