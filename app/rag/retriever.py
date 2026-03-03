"""RAG 检索器 - 将用户查询转换为向量并在 Milvus 中检索相关文档"""

import logging

from app.config import get_settings
from app.core.embedding import embed_query
from app.core.errors import EmbeddingError, MilvusError
from app.storage.milvus_client import search_vectors

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    dataset_ids: list[str],
    top_k: int = 0,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    检索与查询相关的文档块。

    流程: query → embed → Milvus search → filter by threshold

    Args:
        query: 用户查询文本
        dataset_ids: 要搜索的知识库 dataset ID 列表
        top_k: 返回的最大结果数 (0 = 使用配置默认值)
        score_threshold: 最低相似度阈值 (0 = 使用配置默认值)

    Returns:
        检索结果列表 [{text, source_name, source_type, score, collection_id, chunk_index}]
    """
    if not query.strip():
        logger.warning("Empty query, returning no results")
        return []

    if not dataset_ids:
        logger.warning("No dataset_ids provided, returning no results")
        return []

    settings = get_settings()
    if top_k <= 0:
        top_k = settings.TOP_K
    if score_threshold <= 0:
        score_threshold = settings.SIMILARITY_THRESHOLD

    logger.info(
        f"Retrieving: query_len={len(query)}, datasets={len(dataset_ids)}, "
        f"top_k={top_k}, threshold={score_threshold}"
    )

    try:
        # 1. 将查询文本转换为向量
        query_embedding = embed_query(query)

        # 2. 在 Milvus 中搜索
        results = search_vectors(
            query_embedding=query_embedding,
            dataset_ids=dataset_ids,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        logger.info(f"Retrieved {len(results)} chunks above threshold {score_threshold}")
        return results

    except EmbeddingError as e:
        logger.error(f"Embedding failed during retrieval: {e}")
        raise
    except MilvusError as e:
        logger.error(f"Milvus search failed during retrieval: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during retrieval: {e}")
        raise
