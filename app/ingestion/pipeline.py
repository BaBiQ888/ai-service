"""数据摄入 Pipeline - 编排: 解析 → 分块 → 向量化 → 存储"""

import logging

from app.core.errors import IngestionError
from app.ingestion.chunker import split_text
from app.ingestion.embedder import embed_chunks
from app.ingestion.parsers.text_parser import parse_text
from app.ingestion.parsers.file_parser import parse_file
from app.storage.milvus_client import insert_vectors

logger = logging.getLogger(__name__)


def ingest_text(
    dataset_id: str,
    collection_id: str,
    name: str,
    text: str,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> dict:
    """
    文本摄入 pipeline: 解析 → 分块 → 向量化 → 存入 Milvus。

    Args:
        dataset_id: 知识库 dataset ID
        collection_id: collection ID
        name: 集合名称
        text: 原始文本内容
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        {"collection_id": str, "chunk_count": int, "status": str}
    """
    logger.info(
        f"Starting text ingestion: dataset={dataset_id}, "
        f"collection={collection_id}, name={name}, text_len={len(text)}"
    )

    try:
        # 1. 解析文本
        cleaned = parse_text(text)
        if not cleaned:
            raise IngestionError("Text is empty after parsing")

        # 2. 分块
        chunks = split_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise IngestionError("No chunks generated from text")

        logger.info(f"Generated {len(chunks)} chunks from text")

        # 3. 向量化
        embeddings = embed_chunks(chunks)

        # 4. 存入 Milvus
        count = insert_vectors(
            dataset_id=dataset_id,
            collection_id=collection_id,
            texts=chunks,
            embeddings=embeddings,
            source_type="text",
            source_name=name,
        )

        logger.info(
            f"Text ingestion complete: collection={collection_id}, chunks={count}"
        )
        return {
            "collection_id": collection_id,
            "chunk_count": count,
            "status": "ready",
        }

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Text ingestion failed: {e}") from e


def ingest_file(
    dataset_id: str,
    collection_id: str,
    file_data: bytes,
    filename: str,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> dict:
    """
    文件摄入 pipeline: 解析文件 → 分块 → 向量化 → 存入 Milvus。

    Args:
        dataset_id: 知识库 dataset ID
        collection_id: collection ID
        file_data: 文件二进制数据
        filename: 文件名
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        {"collection_id": str, "chunk_count": int, "status": str}
    """
    logger.info(
        f"Starting file ingestion: dataset={dataset_id}, "
        f"collection={collection_id}, filename={filename}, size={len(file_data)}"
    )

    try:
        # 1. 解析文件为文本
        text = parse_file(file_data, filename)
        if not text:
            raise IngestionError(f"No text extracted from file: {filename}")

        # 2. 分块
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise IngestionError("No chunks generated from file")

        logger.info(f"Generated {len(chunks)} chunks from file {filename}")

        # 3. 向量化
        embeddings = embed_chunks(chunks)

        # 4. 存入 Milvus
        count = insert_vectors(
            dataset_id=dataset_id,
            collection_id=collection_id,
            texts=chunks,
            embeddings=embeddings,
            source_type="file",
            source_name=filename,
        )

        logger.info(
            f"File ingestion complete: collection={collection_id}, chunks={count}"
        )
        return {
            "collection_id": collection_id,
            "chunk_count": count,
            "status": "ready",
        }

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"File ingestion failed: {e}") from e


def ingest_link(
    dataset_id: str,
    collection_id: str,
    url: str,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> dict:
    """
    链接摄入 pipeline: 爬取网页 → 提取内容 → 分块 → 向量化 → 存入 Milvus。

    Args:
        dataset_id: 知识库 dataset ID
        collection_id: collection ID
        url: 网页链接
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        {"collection_id": str, "chunk_count": int, "status": str}
    """
    logger.info(
        f"Starting link ingestion: dataset={dataset_id}, "
        f"collection={collection_id}, url={url}"
    )

    try:
        # 1. 爬取并解析网页
        from app.ingestion.parsers.link_parser import parse_link

        text = parse_link(url)
        if not text:
            raise IngestionError(f"No content extracted from URL: {url}")

        # 2. 分块
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise IngestionError("No chunks generated from link content")

        logger.info(f"Generated {len(chunks)} chunks from {url}")

        # 3. 向量化
        embeddings = embed_chunks(chunks)

        # 4. 存入 Milvus
        count = insert_vectors(
            dataset_id=dataset_id,
            collection_id=collection_id,
            texts=chunks,
            embeddings=embeddings,
            source_type="link",
            source_name=url,
        )

        logger.info(
            f"Link ingestion complete: collection={collection_id}, chunks={count}"
        )
        return {
            "collection_id": collection_id,
            "chunk_count": count,
            "status": "ready",
        }

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Link ingestion failed: {e}") from e
