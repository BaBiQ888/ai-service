from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: str = ""

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_DB: str = "default"

    # OpenAI Embedding
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # LLM
    LLM_PROVIDER: str = "claude"  # claude / openai / custom
    ANTHROPIC_API_KEY: str = ""
    OPENAI_LLM_API_KEY: str = ""
    CUSTOM_LLM_BASE_URL: str = ""
    CUSTOM_LLM_API_KEY: str = ""
    CUSTOM_LLM_MODEL: str = ""

    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # 文件限制
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
