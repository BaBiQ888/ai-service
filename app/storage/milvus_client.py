"""Milvus 客户端 - 连接管理和 CRUD 操作"""

import logging
import time
import uuid
from typing import Optional

from pymilvus import (
    Collection,
    MilvusClient,
    connections,
    utility,
)

from app.config import get_settings
from app.core.errors import MilvusError
from app.storage.milvus_schema import (
    COLLECTION_NAME,
    INDEX_PARAMS,
    SCHEMA,
    SEARCH_PARAMS,
)

logger = logging.getLogger(__name__)

# 全局客户端实例
_client: Optional[MilvusClient] = None
_collection: Optional[Collection] = None


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        raise MilvusError("Milvus client not initialized. Call init_milvus() first.")
    return _client


def get_collection() -> Collection:
    """获取 kb_vectors collection 实例"""
    global _collection
    if _collection is None:
        raise MilvusError("Milvus collection not initialized. Call init_milvus() first.")
    return _collection


def init_milvus() -> None:
    """初始化 Milvus 连接并确保 collection 存在"""
    global _client, _collection
    settings = get_settings()

    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    logger.info(f"Connecting to Milvus at {uri}")

    try:
        _client = MilvusClient(uri=uri)

        # 使用传统连接方式创建 Collection 对象
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )

        # 确保 collection 存在
        if not utility.has_collection(COLLECTION_NAME):
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            _collection = Collection(
                name=COLLECTION_NAME,
                schema=SCHEMA,
            )
            # 创建索引
            _collection.create_index(**INDEX_PARAMS)
            logger.info(f"Created index on {COLLECTION_NAME}")
        else:
            _collection = Collection(name=COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} already exists")

        # 加载 collection 到内存
        _collection.load()
        logger.info(f"Collection {COLLECTION_NAME} loaded")

    except Exception as e:
        _client = None
        _collection = None
        raise MilvusError(f"Failed to initialize Milvus: {e}") from e


def close_milvus() -> None:
    """关闭 Milvus 连接"""
    global _client, _collection
    try:
        if _collection is not None:
            _collection.release()
        connections.disconnect("default")
        _client = None
        _collection = None
        logger.info("Milvus connection closed")
    except Exception as e:
        logger.warning(f"Error closing Milvus connection: {e}")


def insert_vectors(
    dataset_id: str,
    collection_id: str,
    texts: list[str],
    embeddings: list[list[float]],
    source_type: str,
    source_name: str,
) -> int:
    """
    批量插入向量到 Milvus。

    Args:
        dataset_id: 知识库 dataset ID (partition key)
        collection_id: collection ID (分组 key)
        texts: 文本块列表
        embeddings: 对应的向量列表
        source_type: 来源类型 (text/link/file)
        source_name: 来源名称

    Returns:
        插入的向量数量
    """
    if len(texts) != len(embeddings):
        raise MilvusError(
            f"texts length ({len(texts)}) != embeddings length ({len(embeddings)})"
        )

    if not texts:
        return 0

    now = int(time.time())
    data = [
        {
            "id": str(uuid.uuid4()),
            "dataset_id": dataset_id,
            "collection_id": collection_id,
            "chunk_index": i,
            "text": text[:16384],  # 截断以符合 VARCHAR 限制
            "source_type": source_type,
            "source_name": source_name[:512],
            "embedding": embedding,
            "created_at": now,
        }
        for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    try:
        client = get_milvus_client()
        result = client.insert(
            collection_name=COLLECTION_NAME,
            data=data,
        )
        count = result.get("insert_count", len(data))
        logger.info(
            f"Inserted {count} vectors: dataset={dataset_id}, "
            f"collection={collection_id}, source={source_name}"
        )
        return count
    except Exception as e:
        raise MilvusError(f"Failed to insert vectors: {e}") from e


def search_vectors(
    query_embedding: list[float],
    dataset_ids: list[str],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    在 Milvus 中搜索相似向量。

    Args:
        query_embedding: 查询向量
        dataset_ids: 限定搜索的 dataset IDs
        top_k: 返回的最大结果数
        score_threshold: 最低相似度阈值

    Returns:
        搜索结果列表 [{text, source_name, source_type, score, collection_id, chunk_index}]
    """
    if not dataset_ids:
        return []

    # 构建过滤表达式
    ids_str = ", ".join(f'"{did}"' for did in dataset_ids)
    filter_expr = f"dataset_id in [{ids_str}]"

    try:
        client = get_milvus_client()
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            anns_field="embedding",
            search_params=SEARCH_PARAMS,
            limit=top_k,
            filter=filter_expr,
            output_fields=[
                "text",
                "source_name",
                "source_type",
                "collection_id",
                "chunk_index",
            ],
        )

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                score = hit.get("distance", 0.0)
                if score < score_threshold:
                    continue
                entity = hit.get("entity", {})
                hits.append(
                    {
                        "text": entity.get("text", ""),
                        "source_name": entity.get("source_name", ""),
                        "source_type": entity.get("source_type", ""),
                        "collection_id": entity.get("collection_id", ""),
                        "chunk_index": entity.get("chunk_index", 0),
                        "score": score,
                    }
                )

        logger.info(
            f"Search returned {len(hits)} results for {len(dataset_ids)} datasets"
        )
        return hits

    except Exception as e:
        raise MilvusError(f"Failed to search vectors: {e}") from e


def delete_by_collection_id(collection_id: str) -> int:
    """删除指定 collection_id 的所有向量"""
    try:
        client = get_milvus_client()
        result = client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'collection_id == "{collection_id}"',
        )
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info(f"Deleted vectors for collection_id={collection_id}, count={count}")
        return count
    except Exception as e:
        raise MilvusError(f"Failed to delete vectors: {e}") from e


def delete_by_dataset_id(dataset_id: str) -> int:
    """删除指定 dataset_id 的所有向量"""
    try:
        client = get_milvus_client()
        result = client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'dataset_id == "{dataset_id}"',
        )
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info(f"Deleted vectors for dataset_id={dataset_id}, count={count}")
        return count
    except Exception as e:
        raise MilvusError(f"Failed to delete vectors: {e}") from e


def count_by_dataset_id(dataset_id: str) -> int:
    """统计指定 dataset_id 的向量数量"""
    try:
        collection = get_collection()
        result = collection.query(
            expr=f'dataset_id == "{dataset_id}"',
            output_fields=["count(*)"],
        )
        if result:
            return result[0].get("count(*)", 0)
        return 0
    except Exception as e:
        raise MilvusError(f"Failed to count vectors: {e}") from e


def count_by_collection_id(collection_id: str) -> int:
    """统计指定 collection_id 的向量数量"""
    try:
        collection = get_collection()
        result = collection.query(
            expr=f'collection_id == "{collection_id}"',
            output_fields=["count(*)"],
        )
        if result:
            return result[0].get("count(*)", 0)
        return 0
    except Exception as e:
        raise MilvusError(f"Failed to count vectors: {e}") from e


def check_connection() -> bool:
    """检查 Milvus 连接是否正常"""
    try:
        client = get_milvus_client()
        client.list_collections()
        return True
    except Exception:
        return False
