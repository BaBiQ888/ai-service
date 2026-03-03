import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    """存活检查"""
    return {"status": "ok"}


@router.get("/readiness")
async def readiness():
    """就绪检查 - 验证 Milvus 和 Embedding 连通性"""
    checks = {
        "milvus": False,
        "embedding": False,
    }

    # 检查 Milvus 连接
    try:
        from app.storage.milvus_client import check_connection
        checks["milvus"] = check_connection()
    except Exception as e:
        logger.error(f"Milvus check failed: {e}")

    # 检查 Embedding API
    try:
        from app.core.embedding import check_embedding
        checks["embedding"] = check_embedding()
    except Exception as e:
        logger.error(f"Embedding check failed: {e}")

    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }
