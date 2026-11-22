"""
【数据预处理模块 (Preprocess)】
负责 RAG 链路的“数据摄入 (Data Ingestion)”阶段。
包含：多种格式文件解析器 (parsers，支持 PDF/Word/CSV 等) 与 智能文本切分器 (chunker，包含重叠截断及表头继承策略)。
"""