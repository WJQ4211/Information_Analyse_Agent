"""
情报分析智能体 - 数据采集与知识图谱构建管道 - 数据模型
Intelligence Analysis Agent - Ingestion Pipeline Data Models

定义数据采集、实体抽取、知识图谱入库过程中的中间数据结构。
这些数据模型仅在管道内部使用，入库时转换为 core 模块中的 GraphNode/GraphEdge。
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# =============================================================================
# 枚举类型 (Enums)
# =============================================================================

class DocumentType(Enum):
    """文档类型枚举"""
    WEB_PAGE = "web_page"        # 网页
    PDF = "pdf"                  # PDF 文档
    RSS_FEED = "rss_feed"        # RSS/Atom 订阅源
    PLAIN_TEXT = "plain_text"    # 纯文本


class IngestionStatus(Enum):
    """入库状态枚举"""
    PENDING = "pending"          # 待处理
    EXTRACTING = "extracting"    # 抽取中
    VALIDATING = "validating"    # 验证中
    INGESTED = "ingested"        # 已入库
    FAILED = "failed"            # 失败


# =============================================================================
# 文档数据模型 (Document Models)
# =============================================================================

@dataclass
class Document:
    """
    原始文档

    代表从 OSINT 来源采集到的一篇原始文档（网页、PDF 页面、RSS 条目等）。
    """
    url: str                                    # 来源 URL 或文件路径
    title: str                                  # 文档标题
    content: str                                # 文档正文内容
    doc_type: DocumentType                      # 文档类型
    source_name: str                            # 来源名称（如 "简氏防务周刊"）
    timestamp: datetime = field(default_factory=datetime.now)  # 采集时间
    metadata: dict[str, Any] = field(default_factory=dict)     # 附加元数据

    @property
    def content_hash(self) -> str:
        """计算内容哈希（用于去重）"""
        import hashlib
        return hashlib.md5(self.content.encode("utf-8")).hexdigest()


@dataclass
class DocumentChunk:
    """
    文档分块

    将长文档切分为适合 LLM 处理的块，保留上下文重叠。
    """
    document_url: str       # 所属文档的 URL（外键引用）
    content: str            # 分块内容
    chunk_index: int        # 分块序号（从 0 开始）
    start_pos: int          # 在原文中的起始位置
    end_pos: int            # 在原文中的结束位置


# =============================================================================
# 抽取结果模型 (Extraction Models)
# =============================================================================

@dataclass
class ExtractedEntity:
    """
    抽取的实体

    由 LLM 从原始文本中提取出的结构化实体信息。
    入库时将转换为 core.infrastructure.GraphNode。
    """
    entity_id: str                              # 实体唯一标识
    entity_type: str                            # 实体类型（Weapon, Organization, Person 等）
    name: str                                   # 实体名称
    attributes: dict[str, Any] = field(default_factory=dict)   # 属性键值对
    confidence: float = 1.0                     # 抽取置信度 [0, 1]


@dataclass
class ExtractedRelation:
    """
    抽取的关系

    由 LLM 从原始文本中提取出的实体间关系。
    入库时将转换为 core.infrastructure.GraphEdge。
    """
    source_entity: str      # 源实体 ID 或名称
    target_entity: str      # 目标实体 ID 或名称
    relation_type: str      # 关系类型（developed_by, deployed_at 等）
    confidence: float = 1.0 # 抽取置信度 [0, 1]


@dataclass
class ExtractionResult:
    """
    单次抽取结果

    包含一次 LLM 抽取调用的全部输出。
    """
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    raw_response: str = ""                      # LLM 原始响应文本
    parse_success: bool = True                  # JSON 解析是否成功
    chunk_index: int = 0                        # 对应的文档分块序号


# =============================================================================
# 入库结果模型 (Ingestion Result)
# =============================================================================

@dataclass
class EntityConflict:
    """
    实体冲突记录

    当新抽取的实体与知识图谱中已有事实发生冲突时记录。
    """
    entity_name: str        # 冲突实体名称
    attribute: str          # 冲突属性
    new_value: Any          # 新值
    old_value: Any          # 旧值
    conflict_score: float   # 冲突分数 [0, 1]
    maskirovka_flag: bool   # 是否疑似战略欺骗


@dataclass
class IngestionResult:
    """
    入库结果汇总

    记录一次完整入库操作（单文档或批量）的统计信息。
    """
    document_count: int = 0                     # 处理文档数
    entity_count: int = 0                       # 入库实体数
    relation_count: int = 0                     # 入库关系数
    conflicts: list[EntityConflict] = field(default_factory=list)   # 冲突记录
    warnings: list[str] = field(default_factory=list)               # 警告信息
    status: IngestionStatus = IngestionStatus.INGESTED              # 最终状态
    duration_seconds: float = 0.0               # 耗时（秒）

    @property
    def conflict_count(self) -> int:
        """冲突数量"""
        return len(self.conflicts)

    @property
    def has_maskirovka(self) -> bool:
        """是否存在战略欺骗嫌疑"""
        return any(c.maskirovka_flag for c in self.conflicts)

    def merge(self, other: "IngestionResult") -> None:
        """合并另一个入库结果（用于批量入库）"""
        self.document_count += other.document_count
        self.entity_count += other.entity_count
        self.relation_count += other.relation_count
        self.conflicts.extend(other.conflicts)
        self.warnings.extend(other.warnings)
        self.duration_seconds += other.duration_seconds
