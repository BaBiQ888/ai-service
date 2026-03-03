"""LLM 提供商抽象层 - 支持 Claude / OpenAI / Custom"""

import logging
from typing import Optional

from app.config import get_settings
from app.core.errors import LLMError

logger = logging.getLogger(__name__)


class LLMResponse:
    """LLM 调用结果"""

    def __init__(
        self,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }


def chat_completion(
    messages: list[dict],
    provider: str = "",
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> LLMResponse:
    """
    统一的 LLM 对话接口。

    Args:
        messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
        provider: LLM 提供商 (claude/openai/custom)，空则使用默认
        model: 模型名称，空则使用默认
        temperature: 温度参数
        max_tokens: 最大输出 token 数

    Returns:
        LLMResponse 包含回复内容和 usage 信息
    """
    settings = get_settings()
    provider = provider or settings.LLM_PROVIDER

    logger.info(
        f"LLM chat: provider={provider}, model={model or 'default'}, "
        f"messages={len(messages)}, temp={temperature}"
    )

    if provider == "claude":
        return _call_claude(messages, model, temperature, max_tokens)
    elif provider == "openai":
        return _call_openai(messages, model, temperature, max_tokens)
    elif provider == "custom":
        return _call_custom(messages, model, temperature, max_tokens)
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")


def _call_claude(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """调用 Anthropic Claude API"""
    try:
        import anthropic
    except ImportError:
        raise LLMError(
            "anthropic package not installed. Run: pip install anthropic")

    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not configured")

    model = model or "claude-sonnet-4-20250514"

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    # 分离 system message
    system_content = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    if not user_messages:
        raise LLMError("No user message provided")

    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_content:
            kwargs["system"] = system_content

        response = client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return LLMResponse(
            content=content,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    except anthropic.RateLimitError as e:
        raise LLMError(f"Claude rate limit exceeded: {e}") from e
    except anthropic.APIError as e:
        raise LLMError(f"Claude API error: {e}") from e
    except Exception as e:
        raise LLMError(f"Claude call failed: {e}") from e


def _call_openai(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """调用 OpenAI Chat API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package not installed. Run: pip install openai")

    settings = get_settings()
    api_key = settings.OPENAI_LLM_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise LLMError("OPENAI_LLM_API_KEY is not configured")

    model = model or "gpt-4o"

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    except Exception as e:
        raise LLMError(f"OpenAI call failed: {e}") from e


def _call_custom(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """调用自定义 OpenAI 兼容 API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package not installed. Run: pip install openai")

    settings = get_settings()
    if not settings.CUSTOM_LLM_BASE_URL:
        raise LLMError("CUSTOM_LLM_BASE_URL is not configured")

    model = model or settings.CUSTOM_LLM_MODEL
    if not model:
        raise LLMError("No model specified for custom LLM provider")

    client = OpenAI(
        api_key=settings.CUSTOM_LLM_API_KEY or "no-key",
        base_url=settings.CUSTOM_LLM_BASE_URL,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    except Exception as e:
        raise LLMError(f"Custom LLM call failed: {e}") from e
