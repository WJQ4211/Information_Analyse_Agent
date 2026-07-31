"""
情报分析智能体 - OSINT 知识图谱构建管道
Intelligence Analysis Agent - OSINT Knowledge Graph Ingestion Pipeline

主编排管道，串联数据采集 → 实体抽取 → NATO 6×6 验证 → 交叉验证 → 知识图谱入库。

数据流：
    Document → chunk → EntityExtractor → ExtractedEntity/Relation
      → assess_evidence() → cross_validate_fact()
      → KnowledgeGraph.add_node() / add_edge()
      → TimeSeriesMemory.add()
"""

import time
from datetime import datetime
from typing import Optional

from core.infrastructure import (
    KnowledgeGraph,
    LLMClient,
    TimeSeriesMemory,
    VectorDatabase,
    GraphNode,
    GraphEdge,
)
from core.algorithms import (
    SourceInfo,
    SourceReliability,
    Evidence,
    assess_evidence,
    cross_validate_fact,
    SR_NORM,
)
from ingestion.models import (
    Document,
    DocumentType,
    IngestionStatus,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    IngestionResult,
    EntityConflict,
)
from ingestion.connectors import (
    BaseConnector,
    WebScraperConnector,
    PDFParserConnector,
    FeedParserConnector,
)
from ingestion.extraction import EntityExtractor


# 连接器注册表：文档类型 → 连接器类
_CONNECTOR_REGISTRY: dict[str, type[BaseConnector]] = {
    "web": WebScraperConnector,
    "pdf": PDFParserConnector,
    "feed": FeedParserConnector,
}


class IngestionPipeline:
    """
    OSINT 知识图谱构建管道

    核心职责：
    1. 通过连接器采集原始文档
    2. 使用 LLM 抽取实体和关系
    3. 对抽取结果进行 NATO 6×6 评估
    4. 与已有知识图谱进行交叉验证
    5. 将验证通过的数据入库

    使用示例：
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        pipeline = IngestionPipeline(kg, llm)

        result = await pipeline.ingest_from_urls(
            urls=["https://example.com/defense-news"],
            source_info=SourceInfo(name="Example", reliability=SourceReliability.B),
        )
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_client: LLMClient,
        memory: Optional[TimeSeriesMemory] = None,
        vector_db: Optional[VectorDatabase] = None,
        extractor: Optional[EntityExtractor] = None,
    ):
        """
        Args:
            knowledge_graph: 知识图谱实例
            llm_client: LLM 客户端实例
            memory: 时序记忆库（可选）
            vector_db: 向量数据库（可选）
            extractor: 实体抽取器（可选，默认自动创建）
        """
        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client
        self.memory = memory
        self.vector_db = vector_db
        self.extractor = extractor or EntityExtractor(llm_client)

        # 初始化连接器
        self._connectors: dict[str, BaseConnector] = {
            name: cls() for name, cls in _CONNECTOR_REGISTRY.items()
        }

    # =========================================================================
    # 公共 API
    # =========================================================================

    async def ingest_document(self, document: Document,
                               source_info: SourceInfo) -> IngestionResult:
        """
        入库单篇文档

        完整的管道流程：抽取 → 验证 → 交叉验证 → 入库

        Args:
            document: 待入库的文档
            source_info: 来源信息（用于 NATO 6×6 评级）

        Returns:
            IngestionResult 入库结果
        """
        start_time = time.time()
        result = IngestionResult(document_count=1)

        try:
            # 跳过空文档
            if not document.content or not document.content.strip():
                result.warnings.append(f"文档内容为空: {document.url}")
                result.status = IngestionStatus.FAILED
                return result

            # 步骤 1: 抽取实体和关系
            result.status = IngestionStatus.EXTRACTING
            extraction = await self._step_extract(document)

            if not extraction.entities:
                result.warnings.append(f"未抽取到任何实体: {document.url}")
                result.status = IngestionStatus.FAILED
                result.duration_seconds = time.time() - start_time
                return result

            # 步骤 2-4: 对每个实体进行验证和入库
            result.status = IngestionStatus.VALIDATING
            for entity in extraction.entities:
                entity_result = await self._process_entity(
                    entity, document, source_info
                )
                result.entity_count += entity_result.entity_count
                result.conflicts.extend(entity_result.conflicts)
                result.warnings.extend(entity_result.warnings)

            # 步骤 5: 入库关系
            for relation in extraction.relations:
                rel_ingested = await self._step_ingest_relation(relation)
                result.relation_count += rel_ingested

            # 步骤 6: 存储到时序记忆库
            if self.memory:
                self._store_to_memory(document, extraction)

            # 步骤 7: 存储到向量数据库（支持语义搜索）
            if self.vector_db:
                self._store_to_vector_db(document, extraction)

            result.status = IngestionStatus.INGESTED

        except Exception as e:
            result.status = IngestionStatus.FAILED
            result.warnings.append(f"入库异常: {e}")

        result.duration_seconds = time.time() - start_time
        return result

    async def ingest_from_urls(self, urls: list[str],
                                source_info: SourceInfo) -> IngestionResult:
        """
        从 URL 列表批量入库

        自动根据 URL 类型选择对应的连接器（web/feed）。
        对于 RSS/Feed 源，使用 fetch_entries() 逐条入库以提升抽取质量。

        Args:
            urls: URL 列表
            source_info: 来源信息

        Returns:
            IngestionResult 批量入库结果
        """
        combined = IngestionResult()

        for url in urls:
            connector_type = self._detect_connector_type(url)
            connector = self._connectors.get(connector_type)

            if not connector:
                combined.warnings.append(f"无法识别的连接器类型: {url}")
                continue

            try:
                # RSS/Feed 源：逐条采集，每条新闻独立入库
                if connector_type == "feed" and hasattr(connector, "fetch_entries"):
                    documents = await connector.fetch_entries(url)
                    for document in documents:
                        if source_info.name:
                            document.source_name = source_info.name
                        doc_result = await self.ingest_document(document, source_info)
                        combined.merge(doc_result)
                else:
                    # 普通网页：单文档采集
                    document = await connector.fetch(url)
                    if source_info.name:
                        document.source_name = source_info.name
                    doc_result = await self.ingest_document(document, source_info)
                    combined.merge(doc_result)

            except (ValueError, ConnectionError) as e:
                combined.warnings.append(f"采集失败 [{url}]: {e}")

        return combined

    async def ingest_from_files(self, paths: list[str],
                                 source_info: SourceInfo) -> IngestionResult:
        """
        从本地文件列表批量入库（支持 PDF）

        Args:
            paths: 文件路径列表
            source_info: 来源信息

        Returns:
            IngestionResult 批量入库结果
        """
        combined = IngestionResult()

        for path in paths:
            connector_type = self._detect_file_connector_type(path)
            connector = self._connectors.get(connector_type)

            if not connector:
                combined.warnings.append(f"不支持的文件类型: {path}")
                continue

            try:
                document = await connector.fetch(path)
                if source_info.name:
                    document.source_name = source_info.name

                doc_result = await self.ingest_document(document, source_info)
                combined.merge(doc_result)

            except (ValueError, ConnectionError, ImportError) as e:
                combined.warnings.append(f"文件处理失败 [{path}]: {e}")

        return combined

    # =========================================================================
    # 管道步骤（内部方法）
    # =========================================================================

    async def _step_extract(self, document: Document) -> ExtractionResult:
        """步骤 1: 使用 LLM 从文档中抽取实体和关系"""
        return await self.extractor.extract(document.content, document_url=document.url)

    async def _process_entity(
        self,
        entity: ExtractedEntity,
        document: Document,
        source_info: SourceInfo,
    ) -> IngestionResult:
        """
        处理单个实体：验证 → 交叉验证 → 入库

        Returns:
            IngestionResult 包含该实体的入库结果
        """
        result = IngestionResult()

        # 步骤 2: NATO 6×6 评估
        assessment = self._step_validate(entity, document, source_info)

        # 步骤 3: 交叉验证（与已有图谱事实对比）
        for attr_name, attr_value in entity.attributes.items():
            conflict_result = await self._step_cross_validate(
                entity.name, attr_name, attr_value
            )

            # 从交叉验证结果中提取真实的旧值
            old_value = self._extract_old_value(conflict_result, attr_name)

            if conflict_result.get("maskirovka_flag"):
                result.conflicts.append(EntityConflict(
                    entity_name=entity.name,
                    attribute=attr_name,
                    new_value=attr_value,
                    old_value=old_value,
                    conflict_score=conflict_result.get("conflict_score", 0.0),
                    maskirovka_flag=True,
                ))
            elif conflict_result.get("conflict_score", 0) > 0.3:
                result.conflicts.append(EntityConflict(
                    entity_name=entity.name,
                    attribute=attr_name,
                    new_value=attr_value,
                    old_value=old_value,
                    conflict_score=conflict_result.get("conflict_score", 0.0),
                    maskirovka_flag=False,
                ))

        # 步骤 4: 入库到知识图谱
        self._step_ingest_entity(entity, assessment, document, source_info)
        result.entity_count = 1

        return result

    def _step_validate(
        self,
        entity: ExtractedEntity,
        document: Document,
        source_info: SourceInfo,
    ) -> "AssessmentResult":
        """
        步骤 2: 对抽取的实体进行 NATO 6×6 评估

        将实体视为一条证据，评估其来源可靠度和信息可信度。
        """
        evidence = Evidence(
            content=f"{entity.name}: {entity.attributes}",
            source=source_info,
            timestamp=document.timestamp,
            entity=entity.name,
            cross_references=[document.source_name],
        )

        return assess_evidence(evidence, self.knowledge_graph)

    async def _step_cross_validate(
        self, entity_name: str, attribute: str, value
    ) -> dict:
        """
        步骤 3: 与知识图谱已有事实进行交叉验证

        检测冲突和潜在的战略欺骗（Maskirovka）。
        """
        return cross_validate_fact(
            entity=entity_name,
            attribute=attribute,
            value=value,
            knowledge_graph=self.knowledge_graph,
        )

    def _step_ingest_entity(
        self,
        entity: ExtractedEntity,
        assessment,
        document: Document,
        source_info: SourceInfo,
    ) -> None:
        """
        步骤 4: 将验证实体入库到知识图谱

        置信度计算：LLM 抽取置信度 × 来源可靠度权重
        """
        # 计算调整后的置信度
        source_weight = SR_NORM.get(
            assessment.source_reliability, 0.5
        )
        adjusted_confidence = entity.confidence * source_weight

        # 构建 GraphNode
        node = GraphNode(
            id=entity.entity_id,
            type=entity.entity_type,
            name=entity.name,
            attributes={
                **entity.attributes,
                "ingestion_timestamp": datetime.now().isoformat(),
                "original_confidence": entity.confidence,
                "source_reliability": assessment.source_reliability.value,
            },
            confidence=adjusted_confidence,
            sources=[document.source_name],
        )

        self.knowledge_graph.add_node(node)

    async def _step_ingest_relation(self, relation: ExtractedRelation) -> int:
        """
        步骤 5: 将关系入库到知识图谱

        Returns:
            1 表示入库成功，0 表示跳过
        """
        # 检查源实体和目标实体是否已存在于图谱中
        source_exists = relation.source_entity in self.knowledge_graph.nodes
        target_exists = relation.target_entity in self.knowledge_graph.nodes

        # 如果实体不存在，尝试按名称查找
        if not source_exists:
            source_node = self._find_node_by_name(relation.source_entity)
            if source_node:
                relation.source_entity = source_node.id
                source_exists = True

        if not target_exists:
            target_node = self._find_node_by_name(relation.target_entity)
            if target_node:
                relation.target_entity = target_node.id
                target_exists = True

        # 只有两端都存在时才入库关系
        if not source_exists or not target_exists:
            return 0

        edge = GraphEdge(
            source_id=relation.source_entity,
            target_id=relation.target_entity,
            relation=relation.relation_type,
            confidence=relation.confidence,
        )

        self.knowledge_graph.add_edge(edge)
        return 1

    def _store_to_memory(self, document: Document,
                          extraction: ExtractionResult) -> None:
        """步骤 6: 存储到时序记忆库"""
        if not self.memory:
            return

        # 存储文档级别的信息
        self.memory.add(
            content=f"[{document.source_name}] {document.title}",
            metadata={
                "key": "ingestion",
                "value": document.source_name,
                "type": "document",
                "doc_type": document.doc_type.value,
                "entity_count": len(extraction.entities),
                "relation_count": len(extraction.relations),
                "confidence": sum(e.confidence for e in extraction.entities)
                              / max(1, len(extraction.entities)),
            },
            timestamp=document.timestamp,
        )

        # 存储每个实体的信息
        for entity in extraction.entities:
            self.memory.add(
                content=f"{entity.name} ({entity.entity_type})",
                metadata={
                    "key": "entity",
                    "value": entity.name,
                    "type": entity.entity_type,
                    "entity_id": entity.entity_id,
                    "confidence": entity.confidence,
                },
                timestamp=document.timestamp,
            )

    def _store_to_vector_db(self, document: Document,
                             extraction: ExtractionResult) -> None:
        """
        将抽取结果存储到向量数据库，支持后续的语义相似度搜索

        使用简单的文本特征哈希生成伪向量（用于内存模式）。
        如果 VectorDatabase 配置了真实后端（Qdrant/Milvus），
        则可以使用 sentence-transformers 生成真实嵌入向量。
        """
        if not self.vector_db:
            return

        # 为每个实体存储向量
        for entity in extraction.entities:
            # 构建实体文本描述
            entity_text = f"{entity.name} {entity.entity_type}"
            if entity.attributes:
                attrs_str = " ".join(
                    f"{k}:{v}" for k, v in entity.attributes.items()
                )
                entity_text += f" {attrs_str}"

            # 生成简单特征向量（文本哈希伪向量，适用于内存模式）
            vector = self._text_to_vector(entity_text)

            self.vector_db.store(
                vector_id=entity.entity_id,
                vector=vector,
                metadata={
                    "name": entity.name,
                    "type": entity.entity_type,
                    "entity_id": entity.entity_id,
                    "confidence": entity.confidence,
                    "source": document.source_name,
                    "document_url": document.url,
                    "attributes": entity.attributes,
                },
            )

        # 为文档分块存储向量（支持文档级语义搜索）
        chunks = self.extractor.chunk_text(
            document.content, document_url=document.url
        )
        for chunk in chunks:
            if chunk.content.strip():
                chunk_id = f"{document.content_hash}_chunk_{chunk.chunk_index}"
                vector = self._text_to_vector(chunk.content)
                self.vector_db.store(
                    vector_id=chunk_id,
                    vector=vector,
                    metadata={
                        "type": "document_chunk",
                        "document_url": document.url,
                        "source": document.source_name,
                        "chunk_index": chunk.chunk_index,
                        "content_preview": chunk.content[:200],
                    },
                )

    @staticmethod
    def _text_to_vector(text: str, dim: int = 128) -> list[float]:
        """
        将文本转换为简单特征向量（内存模式的伪向量）

        使用字符 n-gram 哈希生成固定维度的稀疏向量。
        这不是真正的语义嵌入，仅用于内存模式下的基本相似度搜索。
        生产环境应使用 sentence-transformers 生成真实嵌入。
        """
        import hashlib
        vector = [0.0] * dim

        # 使用 2-gram 和 3-gram 哈希填充向量
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                ngram = text[i:i + n]
                h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                vector[idx] += 1.0

        # L2 归一化
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    # =========================================================================
    # 辅助方法
    # =========================================================================

    @staticmethod
    def _extract_old_value(conflict_result: dict, attribute: str):
        """
        从交叉验证结果中提取历史旧值

        Args:
            conflict_result: cross_validate_fact 返回的结果字典
            attribute: 属性名

        Returns:
            旧值，如果无法提取则返回 "未知"
        """
        supporting = conflict_result.get("supporting_evidence", [])
        if supporting:
            # 取第一条支撑证据的值作为旧值
            for evidence in supporting:
                val = evidence.get("value")
                if val is not None:
                    return val

        # 如果没有支撑证据，尝试从图谱查询结果中获取
        return "未知"

    def _find_node_by_name(self, name: str) -> Optional[GraphNode]:
        """按名称在知识图谱中查找节点（使用名称索引加速）"""
        return self.knowledge_graph.find_node_by_name(name)

    @staticmethod
    def _detect_connector_type(url: str) -> str:
        """
        根据 URL 特征检测应使用的连接器类型

        web: 普通网页
        feed: RSS/Atom 订阅源（包含 rss/feed/atom 关键词）
        """
        url_lower = url.lower()
        if any(kw in url_lower for kw in ("rss", "feed", "atom", ".xml")):
            return "feed"
        return "web"

    @staticmethod
    def _detect_file_connector_type(path: str) -> str:
        """根据文件扩展名检测连接器类型"""
        if path.lower().endswith(".pdf"):
            return "pdf"
        return ""  # 暂不支持其他文件类型
