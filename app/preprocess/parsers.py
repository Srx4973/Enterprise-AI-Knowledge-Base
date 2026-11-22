"""
【文件解析模块】
负责将 PDF、Word、TXT、CSV 等不同格式的文件，统一解析为带有位置标签(location)的文本块。
"""

import pymupdf
import pandas as pd
from docx import Document
from docx.table import Table
from docx.document import Document as _Doc
from docx.text.paragraph import Paragraph

#=============== 解析文件函数 =========================
# ---------- 1. 按页读取 PDF ----------
def load_pdf_pages(path):
    doc = pymupdf.open(path)
    pages = []
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 提取纯文本，PyMuPDF 对中文支持极好
            text = page.get_text("text")
            if text:
                pages.append((f"第 {page_num + 1} 页", text))
    finally:
          doc.close()
    return pages

# ----------按段落号来定位读取txt -----------
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pages = []
    buffer = []
    start_line = 1

    for i, line in enumerate(lines, start=1):
        if line.strip() == "":                    # 空行 = 段落边界
            if buffer:
                text = "".join(buffer)
                if text.strip():
                    pages.append((f"第 {start_line}-{i-1} 行", text))
                buffer = []
            start_line = i + 1
        else:
            if not buffer:
                start_line = i
            buffer.append(line)

    if buffer:                                     # 文件末尾没空行
        text = "".join(buffer)
        if text.strip():
            pages.append((f"第 {start_line}-{len(lines)} 行", text))

    return pages

# ----------按段落标签来定位读取word -----------
def _iter_block_items(parent):
    """按文档实际顺序遍历段落和表格"""
    if isinstance(parent, _Doc):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc
    for child in parent_elm.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('}tbl'):
            yield Table(child, parent)

def load_docx(path):
    doc = Document(path)
    pages = []
    idx = 0

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            if block.text.strip():
                idx += 1
                pages.append((f"第 {idx} 块(段落)", block.text))
        elif isinstance(block, Table):
            table_text = ""
            for row in block.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_text += " | ".join(row_data) + "\n"
            if table_text.strip():
                idx += 1
                pages.append((f"第 {idx} 块(表格)", table_text))

    return pages

# ----------按行来定位读取csv/excel -----------
def dataframe_to_rows(df):
    df = df.fillna("")
    rows = []

    for i, (index, row) in enumerate(df.iterrows()): #逐行遍历，.iterrows()返回index 是行号，row 是这一行的数据
        parts = [
            f"{col}: {val}"              #3.把列名和值拼成 "列名: 值"
            for col, val in row.items()  #1.遍历每一列的 (列名, 值)
            if str(val).strip()          #2.过滤：值转成字符串后去掉空格，非空才保留
        ]
        row_text = "；".join(parts)
        rows.append((f"第 {i + 2} 行", row_text))

    return rows

def load_csv(path):
    return dataframe_to_rows(pd.read_csv(path))


def load_excel(path):
    return dataframe_to_rows(pd.read_excel(path))
