"""网页链接解析器 - 爬取网页并提取文本内容

使用 requests + BeautifulSoup 爬取并解析网页。
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

from app.core.errors import ParsingError

logger = logging.getLogger(__name__)

# 请求超时（秒）
REQUEST_TIMEOUT = 30

# 请求头
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
}

# 需要移除的标签
REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "iframe", "svg",
]


def parse_link(url: str) -> str:
    """
    爬取网页并提取纯文本内容。

    Args:
        url: 网页 URL

    Returns:
        提取的文本内容

    Raises:
        ParsingError: 爬取或解析失败
    """
    logger.info(f"Fetching URL: {url}")

    try:
        resp = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise ParsingError(f"Request timed out after {REQUEST_TIMEOUT}s: {url}")
    except requests.exceptions.ConnectionError:
        raise ParsingError(f"Failed to connect to URL: {url}")
    except requests.exceptions.HTTPError as e:
        raise ParsingError(f"HTTP error {resp.status_code}: {url}") from e
    except requests.exceptions.RequestException as e:
        raise ParsingError(f"Failed to fetch URL: {e}") from e

    # 检测编码
    content_type = resp.headers.get("Content-Type", "")
    if "charset" not in content_type:
        resp.encoding = resp.apparent_encoding or "utf-8"

    html = resp.text
    if not html.strip():
        raise ParsingError(f"Empty response from URL: {url}")

    logger.info(f"Fetched {len(html)} chars from {url}")

    # 解析 HTML
    text = _extract_text(html)
    text = text.strip()

    if not text:
        raise ParsingError(f"No text content extracted from URL: {url}")

    logger.info(f"Extracted {len(text)} chars from {url}")
    return text


def _extract_text(html: str) -> str:
    """从 HTML 中提取纯文本"""
    soup = BeautifulSoup(html, "html.parser")

    # 移除不需要的标签
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 优先提取 <article> 或 <main> 内容
    main_content = soup.find("article") or soup.find("main")
    if main_content:
        text = main_content.get_text(separator="\n")
    else:
        # fallback: 提取 body
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

    # 清理文本
    text = _clean_text(text)
    return text


def _clean_text(text: str) -> str:
    """清理提取的文本"""
    # 移除多余空行（保留最多两个连续换行）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 移除行首行尾空白
    lines = [line.strip() for line in text.split("\n")]
    # 移除空行过多的情况
    cleaned_lines = []
    empty_count = 0
    for line in lines:
        if not line:
            empty_count += 1
            if empty_count <= 1:
                cleaned_lines.append("")
        else:
            empty_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
