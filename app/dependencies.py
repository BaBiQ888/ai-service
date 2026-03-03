import logging
from functools import wraps

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """验证 API Key"""
    if not settings.API_KEY:
        return "no-auth"
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key
