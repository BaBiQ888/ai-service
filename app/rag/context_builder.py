"""上下文构建器 - 将检索结果组装为 LLM 可用的上下文"""

import logging

logger = logging.getLogger(__name__)

# 上下文模板
CONTEXT_TEMPLATE = """以下是从知识库中检索到的相关信息，请基于这些信息回答用户问题。
如果信息不足以回答，请如实说明。

---
{chunks}
---"""

CHUNK_TEMPLATE = "[来源: {source_name} | 相关度: {score:.2f}]\n{text}"


def build_context(
    retrieved_chunks: list[dict],
    max_context_tokens: int = 4000,
) -> tuple[str, list[dict]]:
    """
    将检索到的文档块组装为上下文字符串。

    Args:
        retrieved_chunks: 检索结果列表，每个元素包含 text, source_name, score 等
        max_context_tokens: 上下文最大 token 数（粗略按字符数 / 2 估算）

    Returns:
        (context_str, sources) - 上下文字符串和来源列表
    """
    if not retrieved_chunks:
        return "", []

    # 按相关度排序（高→低）
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)

    # 组装上下文块
    context_parts = []
    sources = []
    total_chars = 0
    max_chars = max_context_tokens * 2  # 粗略估算: 1 token ≈ 2 chars (中英混合)

    for chunk in sorted_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        formatted = CHUNK_TEMPLATE.format(
            source_name=chunk.get("source_name", "unknown"),
            score=chunk.get("score", 0),
            text=text,
        )

        # 检查是否超出长度限制
        if total_chars + len(formatted) > max_chars:
            # 尝试截断最后一块
            remaining = max_chars - total_chars
            if remaining > 200:  # 至少保留200字符才值得加入
                formatted = formatted[:remaining] + "..."
                context_parts.append(formatted)
                sources.append(_make_source(chunk))
            break

        context_parts.append(formatted)
        total_chars += len(formatted)

        sources.append(_make_source(chunk))

    if not context_parts:
        return "", []

    chunks_text = "\n\n".join(context_parts)
    context = CONTEXT_TEMPLATE.format(chunks=chunks_text)

    logger.info(
        f"Built context: {len(context_parts)} chunks, "
        f"{len(context)} chars, {len(sources)} sources"
    )
    return context, sources


def build_messages(
    system_prompt: str,
    context: str,
    user_message: str,
) -> list[dict]:
    """
    构建发送给 LLM 的消息列表。

    Args:
        system_prompt: 系统 prompt (角色定义等)
        context: 检索上下文
        user_message: 用户消息

    Returns:
        消息列表 [{"role": "system/user", "content": "..."}]
    """
    messages = []

    # System prompt
    if system_prompt or context:
        system_content = ""
        if system_prompt:
            system_content = system_prompt
        if context:
            if system_content:
                system_content += "\n\n"
            system_content += context
        messages.append({"role": "system", "content": system_content})

    # User message
    messages.append({"role": "user", "content": user_message})

    return messages


def _make_source(chunk: dict) -> dict:
    """构建来源信息"""
    return {
        "chunk_text": chunk.get("text", "")[:200],
        "source_name": chunk.get("source_name", ""),
        "source_type": chunk.get("source_type", ""),
        "score": round(chunk.get("score", 0), 4),
    }
