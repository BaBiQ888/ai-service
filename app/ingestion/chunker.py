"""文本分块器 - 基于 LangChain RecursiveCharacterTextSplitter"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


def split_text(
    text: str,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> list[str]:
    """
    将文本分割为多个块。

    Args:
        text: 输入文本
        chunk_size: 块大小（0 则使用默认配置）
        chunk_overlap: 块重叠（0 则使用默认配置）

    Returns:
        文本块列表
    """
    if not text.strip():
        return []

    settings = get_settings()
    size = chunk_size if chunk_size > 0 else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap > 0 else settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", "！",
                    "!", "？", "?", "；", ";", " ", ""],
    )

    chunks = splitter.split_text(text)
    # 过滤空块
    chunks = [c.strip() for c in chunks if c.strip()]

    logger.info(
        f"Split text ({len(text)} chars) into {len(chunks)} chunks "
        f"(size={size}, overlap={overlap})"
    )
    return chunks
