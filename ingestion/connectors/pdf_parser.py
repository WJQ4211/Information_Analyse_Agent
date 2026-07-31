"""
情报分析智能体 - PDF 解析连接器
Intelligence Analysis Agent - PDF Parser Connector

使用 PyMuPDF (fitz) 从 PDF 文档中提取文本内容。
"""

import os
import asyncio
from datetime import datetime

from ingestion.models import Document, DocumentType
from ingestion.connectors.base import BaseConnector


class PDFParserConnector(BaseConnector):
    """
    PDF 解析连接器

    从本地 PDF 文件中提取文本内容，支持：
    - 按页提取文本
    - 提取 PDF 元数据（标题、作者、创建日期）
    - 多页文本合并
    """

    async def fetch(self, source: str) -> Document:
        """从 PDF 文件路径解析文档"""
        if not self.validate_source(source):
            raise ValueError(f"无效的 PDF 文件路径: {source}")

        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "PDF 解析需要安装 pymupdf: pip install pymupdf>=1.23.0"
            )

        # 在线程中执行阻塞的 PDF I/O
        try:
            result = await asyncio.to_thread(self._parse_pdf_sync, source, fitz)
        except ConnectionError:
            raise
        except Exception as e:
            raise ConnectionError(f"PDF 处理失败 [{source}]: {e}") from e

        return result

    @staticmethod
    def _parse_pdf_sync(source: str, fitz) -> Document:
        """同步解析 PDF（在线程池中运行）"""
        try:
            doc = fitz.open(source)
        except Exception as e:
            raise ConnectionError(f"PDF 打开失败 [{source}]: {e}") from e

        try:
            # 提取元数据
            pdf_meta = doc.metadata or {}
            title = pdf_meta.get("title") or os.path.basename(source)

            # 按页提取文本
            page_texts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    page_texts.append(text)

            # 合并全文
            full_content = "\n\n".join(page_texts)

            # 提取创建日期
            timestamp = PDFParserConnector._parse_pdf_date(
                pdf_meta.get("creationDate", "")
            )

            return Document(
                url=source,
                title=title,
                content=full_content,
                doc_type=DocumentType.PDF,
                source_name=os.path.basename(source),
                timestamp=timestamp,
                metadata={
                    "page_count": len(page_texts),
                    "author": pdf_meta.get("author", ""),
                    "subject": pdf_meta.get("subject", ""),
                    "keywords": pdf_meta.get("keywords", ""),
                    "total_characters": len(full_content),
                },
            )
        finally:
            doc.close()

    def validate_source(self, source: str) -> bool:
        """验证 PDF 文件路径是否有效"""
        if not source:
            return False
        # 必须是 .pdf 扩展名
        if not source.lower().endswith(".pdf"):
            return False
        # 文件必须存在
        if not os.path.isfile(source):
            return False
        return True

    @staticmethod
    def _parse_pdf_date(date_str: str) -> datetime:
        """
        解析 PDF 日期格式

        PDF 日期格式通常为：D:YYYYMMDDHHmmss+HH'mm'
        """
        if not date_str:
            return datetime.now()

        try:
            # 移除 "D:" 前缀
            cleaned = date_str.replace("D:", "")
            # 取前 14 个字符（YYYYMMDDHHmmss）
            if len(cleaned) >= 14:
                return datetime.strptime(cleaned[:14], "%Y%m%d%H%M%S")
            elif len(cleaned) >= 8:
                return datetime.strptime(cleaned[:8], "%Y%m%d")
        except ValueError:
            pass

        return datetime.now()
