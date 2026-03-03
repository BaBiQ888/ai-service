"""纯文本解析器"""

import logging

logger = logging.getLogger(__name__)


def parse_text(text: str) -> str:
    """
    解析纯文本内容。清理和标准化。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    # 标准化换行符
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # 去除首尾空白
    cleaned = cleaned.strip()

    logger.info(f"Parsed text: {len(text)} -> {len(cleaned)} chars")
    return cleaned
