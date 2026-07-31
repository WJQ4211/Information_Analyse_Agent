"""
情报分析智能体 - 数据采集与知识图谱构建管道
Intelligence Analysis Agent - OSINT Ingestion Pipeline

从公开网络（OSINT）中采集情报数据，通过 LLM 抽取实体和关系，
经过 NATO 6×6 评估和交叉验证后入库到知识图谱。

使用示例：
    from ingestion import IngestionPipeline, WebScraperConnector, EntityExtractor
    from core import KnowledgeGraph, LLMClient, LLMConfig
    from core.algorithms import SourceInfo, SourceReliability

    kg = KnowledgeGraph()
    llm = LLMClient(LLMConfig())
    pipeline = IngestionPipeline(kg, llm)

    source = SourceInfo(name="简氏防务", reliability=SourceReliability.A)
    result = await pipeline.ingest_from_urls(
        urls=["https://example.com/defense-article"],
        source_info=source,
    )
"""

# 数据模型
from ingestion.models import (
    DocumentType,
    IngestionStatus,
    Document,
    DocumentChunk,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    IngestionResult,
    EntityConflict,
)

# 数据连接器
from ingestion.connectors import (
    BaseConnector,
    WebScraperConnector,
    PDFParserConnector,
    FeedParserConnector,
)

# 实体抽取
from ingestion.extraction import (
    EntityExtractor,
)

# 主编排管道
from ingestion.pipeline import IngestionPipeline

__all__ = [
    # Models
    "DocumentType",
    "IngestionStatus",
    "Document",
    "DocumentChunk",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "IngestionResult",
    "EntityConflict",

    # Connectors
    "BaseConnector",
    "WebScraperConnector",
    "PDFParserConnector",
    "FeedParserConnector",

    # Extraction
    "EntityExtractor",

    # Pipeline
    "IngestionPipeline",
]
