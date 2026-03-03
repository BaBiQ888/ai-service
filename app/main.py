from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.router import api_router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    settings = get_settings()
    logger.info("Starting AI Service...")
    logger.info(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"LLM provider: {settings.LLM_PROVIDER}")

    # 初始化 Milvus 连接
    from app.storage.milvus_client import init_milvus
    try:
        init_milvus()
        logger.info("Milvus initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Milvus: {e}")
        logger.warning("Service will start but Milvus operations will fail")

    # 初始化 Embedding 服务
    from app.core.embedding import init_embedding
    try:
        init_embedding()
        logger.info("Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        logger.warning("Service will start but embedding operations will fail")

    yield

    # 清理资源
    logger.info("Shutting down AI Service...")
    from app.storage.milvus_client import close_milvus
    close_milvus()


app = FastAPI(
    title="TG-Agent AI Service",
    description="LangChain + Milvus RAG Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)
