"""
情报分析智能体 - 网页采集连接器
Intelligence Analysis Agent - Web Scraper Connector

使用 requests + BeautifulSoup 从公开网页采集情报数据。
"""

import re
import asyncio
import ipaddress
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ingestion.models import Document, DocumentType
from ingestion.connectors.base import BaseConnector


# 常见 User-Agent 列表（轮换使用，降低被封概率）
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# HTTP 请求超时（秒）
_REQUEST_TIMEOUT = 30


class WebScraperConnector(BaseConnector):
    """
    网页采集连接器

    从指定 URL 采集网页正文内容，支持：
    - 自动提取标题、正文、发布日期
    - User-Agent 轮换
    - 超时控制
    - URL 合法性验证（防 SSRF）
    """

    def __init__(self, timeout: int = _REQUEST_TIMEOUT,
                 user_agent_index: int = 0):
        self.timeout = timeout
        self._ua_index = user_agent_index

    async def fetch(self, source: str) -> Document:
        """从 URL 采集网页内容（非阻塞 I/O）"""
        if not self.validate_source(source):
            raise ValueError(f"无效的 URL: {source}")

        headers = {"User-Agent": self._next_user_agent()}

        try:
            # 使用 asyncio.to_thread 避免阻塞事件循环
            response = await asyncio.to_thread(
                requests.get, source, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"网页采集失败 [{source}]: {e}") from e

        # 检测编码
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        # 提取标题
        title = self._extract_title(soup)

        # 提取正文（移除脚本、样式、导航等噪音）
        content = self._extract_content(soup)

        # 提取发布日期
        timestamp = self._extract_timestamp(soup)

        # 提取域名作为来源名称
        parsed = urlparse(source)
        source_name = parsed.netloc

        return Document(
            url=source,
            title=title,
            content=content,
            doc_type=DocumentType.WEB_PAGE,
            source_name=source_name,
            timestamp=timestamp,
            metadata={
                "status_code": response.status_code,
                "content_length": len(content),
                "encoding": response.encoding,
            },
        )

    def validate_source(self, source: str) -> bool:
        """验证 URL 合法性（含 SSRF 防护）"""
        try:
            parsed = urlparse(source)
            # 必须是 HTTP/HTTPS 协议（防 SSRF）
            if parsed.scheme not in ("http", "https"):
                return False
            # 必须有域名
            if not parsed.netloc:
                return False

            hostname = parsed.hostname
            if not hostname:
                return False

            # 拒绝内网地址（完整 SSRF 防护）
            if hostname in ("localhost",):
                return False

            try:
                addr = ipaddress.ip_address(hostname)
                # 拒绝所有私有/保留/环回/链路本地地址
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                    return False
            except ValueError:
                # hostname 不是 IP 地址（是域名），允许通过
                pass

            return True
        except Exception:
            return False

    # ---- 内部方法 ----

    def _next_user_agent(self) -> str:
        """获取下一个 User-Agent（轮换）"""
        ua = _USER_AGENTS[self._ua_index % len(_USER_AGENTS)]
        self._ua_index += 1
        return ua

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """提取网页标题"""
        # 优先取 <h1>，否则取 <title>
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)
        return "无标题"

    @staticmethod
    def _extract_content(soup: BeautifulSoup) -> str:
        """提取网页正文内容"""
        # 移除噪音元素
        noise_tags = ["script", "style", "nav", "header", "footer",
                      "aside", "iframe", "noscript"]
        for tag_name in noise_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 尝试查找 <article> 标签（语义化正文区域）
        article = soup.find("article")
        if article:
            return article.get_text(separator="\n", strip=True)

        # 尝试查找 class 包含 "content" / "article" / "post" 的 div
        content_div = soup.find(
            "div",
            class_=re.compile(r"(content|article|post|main-body)", re.I),
        )
        if content_div:
            return content_div.get_text(separator="\n", strip=True)

        # 回退：取 <body> 全文
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    @staticmethod
    def _extract_timestamp(soup: BeautifulSoup) -> datetime:
        """尝试从网页中提取发布日期"""
        # 查找 <time> 标签
        time_tag = soup.find("time")
        if time_tag:
            dt_attr = time_tag.get("datetime") or time_tag.get("content")
            if dt_attr:
                try:
                    return datetime.fromisoformat(dt_attr.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # 查找 meta 标签中的日期
        for meta in soup.find_all("meta"):
            name = meta.get("name", "") or meta.get("property", "")
            if any(kw in name.lower() for kw in ("date", "time", "published")):
                content = meta.get("content", "")
                try:
                    return datetime.fromisoformat(content.replace("Z", "+00:00"))
                except ValueError:
                    continue

        return datetime.now()
