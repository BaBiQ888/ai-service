from typing import Any, Optional

from pydantic import BaseModel


class APIResponseModel(BaseModel):
    """统一 API 响应格式，与 Go 服务的 APIResponse 保持一致"""
    code: int = 0
    message: str = "success"
    data: Any = None
