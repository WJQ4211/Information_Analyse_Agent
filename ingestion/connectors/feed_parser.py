"""
情报分析智能体 - RSS/Atom 订阅源连接器
Intelligence Analysis Agent - RSS Feed Connector

使用 feedparser 从 RSS/Atom 订阅源批量采集新闻条目。
"""

import asyncio
from datetime import datetime
from time import mktime
from urllib.parse import urlparse

from ingestion.models import Document, DocumentType
from ingestion.connectors.base import BaseConnector


class FeedParserConnector(BaseConnector):
    """
    RSS/Atom 订阅源连接器

    从 RSS/Atom 订阅源批量采集新闻条目，支持：
    - RSS 2.0 / Atom 格式自动检测
    - 条目数量限制
    - 日期过滤
    """

    def __init__(self, max_entries: int = 20):
        """
        Args:
            max_entries: 每次采集的最大条目数
        """
        self.max_entries = max_entries

    async def fetch(self, source: str) -> Document:
        """
        从 RSS/Atom 源采集文档

        注意：RSS 源包含多条新闻，此方法返回包含所有条目的合并文档。
        如需获取独立条目列表，请使用 fetch_entries()。
        """
        entries = await self.fetch_entries(source)
        if not entries:
            return Document(
                url=source,
                title="空订阅源",
                content="",
                doc_type=DocumentType.RSS_FEED,
                source_name=urlparse(source).netloc,
                timestamp=datetime.now(),
                metadata={"entry_count": 0},
            )

        # 合并为单一文档（每条新闻用分隔符分隔）
        combined_content = "\n\n---\n\n".join(
            f"【{e.title}】\n{e.content}" for e in entries
        )

        return Document(
            url=source,
            title=f"RSS 订阅: {entries[0].source_name}",
            content=combined_content,
            doc_type=DocumentType.RSS_FEED,
            source_name=entries[0].source_name,
            timestamp=datetime.now(),
            metadata={"entry_count": len(entries)},
        )

    async def fetch_entries(self, source: str) -> list[Document]:
        """
        从 RSS/Atom 源采集独立条目列表

        使用 asyncio.to_thread 避免阻塞事件循环。

        Returns:
            Document 列表，每个元素代表一条新闻
        """
        if not self.validate_source(source):
            raise ValueError(f"无效的 RSS 源 URL: {source}")

        try:
            import feedparser
        except ImportError:
            raise ImportError(
                "RSS 解析需要安装 feedparser: pip install feedparser>=6.0.0"
            )

        try:
            # 在线程中执行阻塞的 feedparser.parse
            feed = await asyncio.to_thread(feedparser.parse, source)
        except Exception as e:
            raise ConnectionError(f"RSS 解析失败 [{source}]: {e}") from e

        if feed.bozo and not feed.entries:
            raise ConnectionError(
                f"RSS 源解析异常 [{source}]: {feed.bozo_exception}"
            )

        parsed_url = urlparse(source)
        feed_title = feed.feed.get("title", parsed_url.netloc)

        documents = []
        for entry in feed.entries[: self.max_entries]:
            title = entry.get("title", "无标题")
            content = self._extract_entry_content(entry)
            timestamp = self._parse_entry_date(entry)
            link = entry.get("link", source)

            documents.append(Document(
                url=link,
                title=title,
                content=content,
                doc_type=DocumentType.RSS_FEED,
                source_name=feed_title,
                timestamp=timestamp,
                metadata={
                    "feed_title": feed_title,
                    "entry_id": entry.get("id", ""),
                    "author": entry.get("author", ""),
                    "tags": [t.get("term", "") for t in entry.get("tags", [])],
                },
            ))

        return documents

    def validate_source(self, source: str) -> bool:
        """验证 RSS 源 URL 是否合法"""
        try:
            parsed = urlparse(source)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _extract_entry_content(entry: dict) -> str:
        """提取 RSS 条目的正文内容"""
        # 优先使用 summary（摘要），其次 content
        if entry.get("summary"):
            # 清理 HTML 标签
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(entry["summary"], "html.parser")
            return soup.get_text(separator="\n", strip=True)

        if entry.get("content"):
            from bs4 import BeautifulSoup
            content_list = entry["content"]
            text_parts = []
            for c in content_list:
                value = c.get("value", "") if isinstance(c, dict) else str(c)
                soup = BeautifulSoup(value, "html.parser")
                text_parts.append(soup.get_text(separator="\n", strip=True))
            return "\n".join(text_parts)

        return entry.get("description", "")

    @staticmethod
    def _parse_entry_date(entry: dict) -> datetime:
        """解析 RSS 条目发布时间"""
        for date_field in ("published_parsed", "updated_parsed"):
            parsed_time = entry.get(date_field)
            if parsed_time:
                try:
                    return datetime.fromtimestamp(mktime(parsed_time))
                except (ValueError, OSError, OverflowError):
                    continue
        return datetime.now()
