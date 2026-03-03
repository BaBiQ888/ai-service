"""批量 Embedding 封装 - 用于摄入 pipeline"""

import logging

from app.core.embedding import embed_documents
from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


def embed_chunks(texts: list[str]) -> list[list[float]]:
    """
    将文本块列表批量转换为向量。

    Args:
        texts: 文本块列表

    Returns:
        向量列表，与输入一一对应

    Raises:
        EmbeddingError: embedding 失败
    """
    if not texts:
        return []

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embed_documents(texts)

    if len(embeddings) != len(texts):
        raise EmbeddingError(
            f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
        )

    logger.info(f"Successfully embedded {len(texts)} chunks")
    return embeddings
