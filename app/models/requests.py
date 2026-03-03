from typing import Optional

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """RAG 对话请求"""
    message: str
    agent_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    language: str = "English"
    chat_type: str = "private"
    llm_provider: str = ""
    llm_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


class CreateDatasetRequest(BaseModel):
    """创建 dataset 请求"""
    dataset_id: str
    name: str
    description: str = ""


class AddTextCollectionRequest(BaseModel):
    """摄入文本请求"""
    dataset_id: str
    collection_id: str
    name: str
    text: str
    chunk_size: int = 512
    chunk_overlap: int = 50


class AddLinkCollectionRequest(BaseModel):
    """摄入链接请求"""
    dataset_id: str
    collection_id: str
    url: str
    chunk_size: int = 512
    chunk_overlap: int = 50
