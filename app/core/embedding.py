"""OpenAI Embedding 封装 - text-embedding-3-small"""

import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import get_settings
from app.core.errors import EmbeddingError, RateLimitError

logger = logging.getLogger(__name__)

# 全局客户端
_client: Optional[OpenAI] = None

# OpenAI 批量限制
MAX_BATCH_SIZE = 2048


def get_openai_client() -> OpenAI:
    """获取 OpenAI 客户端单例"""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.OPENAI_API_KEY:
            raise EmbeddingError("OPENAI_API_KEY is not configured")
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def init_embedding() -> None:
    """初始化并验证 embedding 服务"""
    try:
        client = get_openai_client()
        settings = get_settings()
        # 用短文本做一次测试调用
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input="test",
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        dim = len(response.data[0].embedding)
        logger.info(
            f"Embedding service ready: model={settings.EMBEDDING_MODEL}, "
            f"dimensions={dim}"
        )
    except Exception as e:
        raise EmbeddingError(
            f"Failed to initialize embedding service: {e}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def embed_query(text: str) -> list[float]:
    """
    将单条文本转换为向量。

    Args:
        text: 输入文本

    Returns:
        1536 维浮点向量
    """
    if not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    settings = get_settings()
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            raise RateLimitError(f"Embedding rate limit hit: {e}") from e
        raise EmbeddingError(f"Failed to embed query: {e}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    批量将文本转换为向量。自动分批处理。

    Args:
        texts: 文本列表

    Returns:
        向量列表，与输入一一对应
    """
    if not texts:
        return []

    settings = get_settings()
    all_embeddings: list[list[float]] = []

    # 按 OpenAI 批量限制分批
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i: i + MAX_BATCH_SIZE]
        # 过滤空文本，记住索引
        non_empty = [(j, t) for j, t in enumerate(batch) if t.strip()]

        if not non_empty:
            # 全是空文本，用零向量填充
            all_embeddings.extend(
                [[0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch)
            )
            continue

        try:
            client = get_openai_client()
            response = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=[t for _, t in non_empty],
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )

            # 构建结果，将空文本位置填零向量
            batch_embeddings = [
                [0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch)
            for idx, emb_data in enumerate(response.data):
                original_idx = non_empty[idx][0]
                batch_embeddings[original_idx] = emb_data.embedding

            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(f"Embedding rate limit hit: {e}") from e
            raise EmbeddingError(
                f"Failed to embed batch {i // MAX_BATCH_SIZE}: {e}"
            ) from e

    logger.info(
        f"Embedded {len(texts)} texts in {(len(texts) - 1) // MAX_BATCH_SIZE + 1} batch(es)")
    return all_embeddings


def check_embedding() -> bool:
    """检查 embedding 服务是否可用"""
    try:
        embed_query("health check")
        return True
    except Exception:
        return False
