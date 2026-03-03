"""自定义异常类"""


class AIServiceError(Exception):
    """AI 服务基础异常"""
    pass


class EmbeddingError(AIServiceError):
    """Embedding API 调用失败"""
    pass


class LLMError(AIServiceError):
    """LLM API 调用失败"""
    pass


class MilvusError(AIServiceError):
    """向量库操作失败"""
    pass


class ParsingError(AIServiceError):
    """文档解析失败"""
    pass


class IngestionError(AIServiceError):
    """数据摄入失败"""
    pass


class RateLimitError(AIServiceError):
    """API 速率限制"""
    pass
