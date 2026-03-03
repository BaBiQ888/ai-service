"""RAG 对话端点 - 检索增强生成"""

import logging

from fastapi import APIRouter, Depends

from app.dependencies import verify_api_key
from app.models.requests import ChatCompletionRequest
from app.models.responses import APIResponseModel
from app.core.errors import EmbeddingError, LLMError, MilvusError
from app.core.llm_provider import chat_completion
from app.rag.retriever import retrieve
from app.rag.context_builder import build_context, build_messages

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=APIResponseModel)
async def chat_completions(
    req: ChatCompletionRequest,
    _api_key: str = Depends(verify_api_key),
):
    """RAG 对话端点: 检索 → 上下文构建 → LLM 生成"""
    logger.info(
        f"Chat request: agent_id={req.agent_id}, message_len={len(req.message)}, "
        f"datasets={len(req.dataset_ids)}, provider={req.llm_provider or 'default'}"
    )

    try:
        # 1. RAG 检索
        sources = []
        context = ""
        if req.dataset_ids:
            retrieved_chunks = retrieve(
                query=req.message,
                dataset_ids=req.dataset_ids,
            )
            context, sources = build_context(retrieved_chunks)
            logger.info(f"RAG retrieved {len(sources)} sources")

        # 2. 构建消息
        messages = build_messages(
            system_prompt=req.system_prompt,
            context=context,
            user_message=req.message,
        )

        # 3. 调用 LLM
        llm_response = chat_completion(
            messages=messages,
            provider=req.llm_provider,
            model=req.llm_model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        logger.info(
            f"Chat complete: tokens={llm_response.total_tokens}, "
            f"sources={len(sources)}"
        )

        return APIResponseModel(
            code=0,
            message="success",
            data={
                "content": llm_response.content,
                "sources": sources,
                "usage": llm_response.to_dict()["usage"],
            },
        )

    except EmbeddingError as e:
        logger.error(f"Embedding error in chat: {e}")
        return APIResponseModel(code=500, message=f"Embedding error: {e}", data=None)
    except MilvusError as e:
        logger.error(f"Milvus error in chat: {e}")
        return APIResponseModel(code=500, message=f"Search error: {e}", data=None)
    except LLMError as e:
        logger.error(f"LLM error in chat: {e}")
        return APIResponseModel(code=500, message=f"LLM error: {e}", data=None)
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        return APIResponseModel(code=500, message=f"Internal error: {e}", data=None)
