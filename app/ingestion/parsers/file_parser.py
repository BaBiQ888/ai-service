"""文件解析器 - 支持 PDF/DOCX/MD/TXT

使用 LangChain document_loaders 解析不同格式的文件。
"""

import logging
import os
import tempfile
from pathlib import Path

from app.core.errors import ParsingError

logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".doc", ".docx", ".md"}


def parse_file(file_data: bytes, filename: str) -> str:
    """
    解析文件内容为纯文本。

    Args:
        file_data: 文件二进制数据
        filename: 文件名（用于判断格式）

    Returns:
        提取的文本内容

    Raises:
        ParsingError: 解析失败
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ParsingError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Parsing file: {filename} ({len(file_data)} bytes, type={ext})")

    # 写入临时文件供 LangChain loader 使用
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)

    try:
        with open(tmp_path, "wb") as f:
            f.write(file_data)

        if ext == ".txt":
            text = _parse_txt(tmp_path)
        elif ext == ".pdf":
            text = _parse_pdf(tmp_path)
        elif ext in (".doc", ".docx"):
            text = _parse_docx(tmp_path)
        elif ext == ".md":
            text = _parse_markdown(tmp_path)
        else:
            raise ParsingError(f"No parser for extension: {ext}")

        text = text.strip()
        if not text:
            raise ParsingError(f"No text content extracted from {filename}")

        logger.info(f"Parsed {filename}: extracted {len(text)} chars")
        return text

    except ParsingError:
        raise
    except Exception as e:
        raise ParsingError(f"Failed to parse {filename}: {e}") from e
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _parse_txt(file_path: str) -> str:
    """解析纯文本文件"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_pdf(file_path: str) -> str:
    """解析 PDF 文件"""
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
    except ImportError:
        # fallback 到 pypdf
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)


def _parse_docx(file_path: str) -> str:
    """解析 Word 文档"""
    try:
        from docx import Document

        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise ParsingError(f"Failed to parse DOCX: {e}") from e


def _parse_markdown(file_path: str) -> str:
    """解析 Markdown 文件（保留原始文本，分块器会处理结构）"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
