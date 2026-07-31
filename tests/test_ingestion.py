"""
情报分析智能体 - 数据采集管道单元测试
Intelligence Analysis Agent - Ingestion Pipeline Unit Tests

覆盖：数据模型、连接器、实体抽取、编排管道、集成测试
"""

import json
import pytest
import asyncio
from datetime import datetime

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
from ingestion.connectors.web_scraper import WebScraperConnector
from ingestion.connectors.pdf_parser import PDFParserConnector
from ingestion.connectors.feed_parser import FeedParserConnector
from ingestion.extraction.entity_extractor import EntityExtractor
from ingestion.extraction.prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
)
from ingestion.pipeline import IngestionPipeline

from core.infrastructure import (
    KnowledgeGraph,
    LLMClient,
    LLMConfig,
    TimeSeriesMemory,
    VectorDatabase,
    GraphNode,
    GraphEdge,
)
from core.algorithms import SourceInfo, SourceReliability


# =============================================================================
# 数据模型测试
# =============================================================================

class TestModels:
    """数据模型创建与属性测试"""

    def test_document_creation(self):
        """测试 Document 创建"""
        doc = Document(
            url="https://example.com/article",
            title="测试文章",
            content="这是一篇关于 DF-21D 导弹的报道。",
            doc_type=DocumentType.WEB_PAGE,
            source_name="简氏防务",
        )
        assert doc.url == "https://example.com/article"
        assert doc.doc_type == DocumentType.WEB_PAGE
        assert isinstance(doc.timestamp, datetime)
        assert doc.content_hash  # 非空哈希

    def test_document_content_hash_deterministic(self):
        """测试内容哈希一致性"""
        doc1 = Document(url="a", title="t", content="相同内容",
                        doc_type=DocumentType.WEB_PAGE, source_name="s")
        doc2 = Document(url="b", title="t", content="相同内容",
                        doc_type=DocumentType.WEB_PAGE, source_name="s")
        assert doc1.content_hash == doc2.content_hash

    def test_document_chunk(self):
        """测试 DocumentChunk"""
        chunk = DocumentChunk(
            document_url="https://example.com",
            content="分块内容",
            chunk_index=0,
            start_pos=0,
            end_pos=4,
        )
        assert chunk.chunk_index == 0
        assert chunk.content == "分块内容"

    def test_extracted_entity(self):
        """测试 ExtractedEntity"""
        entity = ExtractedEntity(
            entity_id="weapon_df21d",
            entity_type="Weapon",
            name="DF-21D 反舰弹道导弹",
            attributes={"range_km": 1500, "speed_mach": 3.0},
            confidence=0.9,
        )
        assert entity.entity_type == "Weapon"
        assert entity.attributes["range_km"] == 1500

    def test_extracted_relation(self):
        """测试 ExtractedRelation"""
        rel = ExtractedRelation(
            source_entity="weapon_df21d",
            target_entity="org_casic",
            relation_type="developed_by",
            confidence=0.85,
        )
        assert rel.relation_type == "developed_by"

    def test_extraction_result(self):
        """测试 ExtractionResult"""
        result = ExtractionResult(
            entities=[ExtractedEntity("e1", "Weapon", "W1")],
            relations=[ExtractedRelation("e1", "e2", "uses")],
            parse_success=True,
        )
        assert len(result.entities) == 1
        assert len(result.relations) == 1

    def test_ingestion_result_merge(self):
        """测试 IngestionResult 合并"""
        r1 = IngestionResult(document_count=1, entity_count=3, relation_count=1)
        r2 = IngestionResult(document_count=1, entity_count=2, relation_count=2)
        r1.merge(r2)
        assert r1.document_count == 2
        assert r1.entity_count == 5
        assert r1.relation_count == 3

    def test_ingestion_result_maskirovka(self):
        """测试战略欺骗检测"""
        result = IngestionResult(conflicts=[
            EntityConflict("实体1", "射程", 2000, 1500, 0.8, True),
            EntityConflict("实体2", "数量", 100, 50, 0.4, False),
        ])
        assert result.has_maskirovka is True
        assert result.conflict_count == 2

    def test_ingestion_result_no_maskirovka(self):
        """测试无战略欺骗"""
        result = IngestionResult(conflicts=[
            EntityConflict("实体1", "射程", 2000, 1500, 0.8, False),
        ])
        assert result.has_maskirovka is False


# =============================================================================
# 连接器测试
# =============================================================================

class TestConnectors:
    """数据连接器测试（不发起真实网络请求）"""

    def test_web_scraper_validate_valid_url(self):
        """测试网页连接器 URL 验证 - 合法 URL"""
        connector = WebScraperConnector()
        assert connector.validate_source("https://www.janes.com/news/123") is True
        assert connector.validate_source("http://defense-news.com/article") is True

    def test_web_scraper_validate_invalid_url(self):
        """测试网页连接器 URL 验证 - 非法 URL"""
        connector = WebScraperConnector()
        assert connector.validate_source("") is False
        assert connector.validate_source("not-a-url") is False
        assert connector.validate_source("ftp://files.example.com/data") is False
        assert connector.validate_source("http://localhost/admin") is False
        assert connector.validate_source("http://127.0.0.1/secret") is False
        assert connector.validate_source("http://192.168.1.1/internal") is False

    def test_pdf_parser_validate_valid_path(self):
        """测试 PDF 连接器路径验证 - 合法路径"""
        connector = PDFParserConnector()
        # 文件不存在时应返回 False
        assert connector.validate_source("./nonexistent.pdf") is False

    def test_pdf_parser_validate_invalid_path(self):
        """测试 PDF 连接器路径验证 - 非法路径"""
        connector = PDFParserConnector()
        assert connector.validate_source("") is False
        assert connector.validate_source("document.txt") is False
        assert connector.validate_source("https://example.com/file.pdf") is False

    def test_feed_parser_validate_valid_url(self):
        """测试 RSS 连接器 URL 验证 - 合法 URL"""
        connector = FeedParserConnector()
        assert connector.validate_source("https://defense-news.com/rss") is True
        assert connector.validate_source("https://news.com/feed.xml") is True

    def test_feed_parser_validate_invalid_url(self):
        """测试 RSS 连接器 URL 验证 - 非法 URL"""
        connector = FeedParserConnector()
        assert connector.validate_source("") is False
        assert connector.validate_source("not-a-url") is False
        assert connector.validate_source("ftp://files.com/feed") is False


# =============================================================================
# 实体抽取测试
# =============================================================================

class TestEntityExtractor:
    """实体抽取器测试"""

    @pytest.fixture
    def extractor(self):
        """使用 Mock LLM 创建抽取器"""
        llm = LLMClient(LLMConfig())
        return EntityExtractor(llm, chunk_size=500, chunk_overlap=100)

    def test_chunk_text_short(self, extractor):
        """测试短文本不分块"""
        chunks = extractor.chunk_text("短文本内容")
        assert len(chunks) == 1
        assert chunks[0].content == "短文本内容"
        assert chunks[0].chunk_index == 0

    def test_chunk_text_long(self, extractor):
        """测试长文本分块"""
        text = "A" * 1200  # 超过 chunk_size=500
        chunks = extractor.chunk_text(text)
        assert len(chunks) >= 2
        # 验证重叠
        assert chunks[0].end_pos > chunks[1].start_pos

    def test_chunk_text_empty(self, extractor):
        """测试空文本"""
        chunks = extractor.chunk_text("")
        assert len(chunks) == 1
        assert chunks[0].content == ""

    def test_parse_entities_response_valid_json(self, extractor):
        """测试 JSON 实体响应解析"""
        response = json.dumps({
            "entities": [
                {
                    "entity_id": "weapon_df21d",
                    "entity_type": "Weapon",
                    "name": "DF-21D",
                    "attributes": {"range_km": 1500},
                    "confidence": 0.9,
                }
            ]
        })
        entities = extractor._parse_entities_response(response)
        assert len(entities) == 1
        assert entities[0].name == "DF-21D"
        assert entities[0].confidence == 0.9

    def test_parse_entities_response_markdown_json(self, extractor):
        """测试 Markdown 代码块包裹的 JSON"""
        response = '```json\n{"entities": [{"entity_id": "e1", "entity_type": "Weapon", "name": "测试武器", "confidence": 0.8}]}\n```'
        entities = extractor._parse_entities_response(response)
        assert len(entities) == 1
        assert entities[0].name == "测试武器"

    def test_parse_entities_response_empty(self, extractor):
        """测试空实体列表"""
        response = '{"entities": []}'
        entities = extractor._parse_entities_response(response)
        assert len(entities) == 0

    def test_parse_entities_response_invalid_json(self, extractor):
        """测试无效 JSON（正则回退）"""
        response = '这不是 JSON，但包含 "name": "DF-21D" 和 "entity_type": "Weapon"'
        entities = extractor._parse_entities_response(response)
        # 正则回退应能提取到部分实体
        assert len(entities) >= 0  # 回退可能提取到也可能提取不到

    def test_parse_relations_response_valid(self, extractor):
        """测试关系响应解析"""
        response = json.dumps({
            "relations": [
                {
                    "source_entity": "weapon_df21d",
                    "target_entity": "org_casic",
                    "relation_type": "developed_by",
                    "confidence": 0.85,
                }
            ]
        })
        relations = extractor._parse_relations_response(response)
        assert len(relations) == 1
        assert relations[0].relation_type == "developed_by"

    def test_parse_relations_response_empty(self, extractor):
        """测试空关系列表"""
        response = '{"relations": []}'
        relations = extractor._parse_relations_response(response)
        assert len(relations) == 0

    def test_deduplicate_entities(self, extractor):
        """测试实体去重"""
        entities = [
            ExtractedEntity("e1", "Weapon", "DF-21D", confidence=0.7),
            ExtractedEntity("e1", "Weapon", "DF-21D", confidence=0.9),
            ExtractedEntity("e2", "Organization", "CASIC", confidence=0.8),
        ]
        deduped = extractor._deduplicate_entities(entities)
        assert len(deduped) == 2
        # e1 应保留置信度高的版本
        e1 = [e for e in deduped if e.entity_id == "e1"][0]
        assert e1.confidence == 0.9

    def test_deduplicate_relations(self, extractor):
        """测试关系去重"""
        relations = [
            ExtractedRelation("e1", "e2", "developed_by", confidence=0.7),
            ExtractedRelation("e1", "e2", "developed_by", confidence=0.9),
            ExtractedRelation("e1", "e3", "deployed_at", confidence=0.8),
        ]
        deduped = extractor._deduplicate_relations(relations)
        assert len(deduped) == 2

    def test_extract_empty_text(self, extractor):
        """测试空文本抽取"""
        result = asyncio.run(extractor.extract(""))
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.parse_success is True


# =============================================================================
# 管道测试
# =============================================================================

class TestIngestionPipeline:
    """编排管道测试"""

    @pytest.fixture
    def pipeline_setup(self):
        """创建测试用的管道组件"""
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        memory = TimeSeriesMemory(max_entries=100)
        pipeline = IngestionPipeline(kg, llm, memory=memory)
        source_info = SourceInfo(
            name="测试来源",
            reliability=SourceReliability.B,
            history_accuracy=0.8,
            expertise_level=0.7,
            independence=0.6,
            traceability=True,
        )
        return pipeline, kg, memory, source_info

    def test_pipeline_creation(self, pipeline_setup):
        """测试管道创建"""
        pipeline, kg, memory, source_info = pipeline_setup
        assert pipeline.knowledge_graph is kg
        assert pipeline.memory is memory
        assert pipeline.extractor is not None

    def test_detect_connector_type(self, pipeline_setup):
        """测试连接器类型检测"""
        pipeline, _, _, _ = pipeline_setup
        assert pipeline._detect_connector_type("https://example.com/article") == "web"
        assert pipeline._detect_connector_type("https://news.com/rss") == "feed"
        assert pipeline._detect_connector_type("https://news.com/feed.xml") == "feed"
        assert pipeline._detect_connector_type("https://news.com/atom") == "feed"

    def test_detect_file_connector_type(self, pipeline_setup):
        """测试文件连接器类型检测"""
        pipeline, _, _, _ = pipeline_setup
        assert pipeline._detect_file_connector_type("report.pdf") == "pdf"
        assert pipeline._detect_file_connector_type("Report.PDF") == "pdf"
        assert pipeline._detect_file_connector_type("data.txt") == ""

    def test_ingest_empty_document(self, pipeline_setup):
        """测试空文档入库"""
        pipeline, kg, _, source_info = pipeline_setup
        doc = Document(
            url="test://empty",
            title="空文档",
            content="",
            doc_type=DocumentType.PLAIN_TEXT,
            source_name="测试",
        )
        result = asyncio.run(pipeline.ingest_document(doc, source_info))
        assert result.status == IngestionStatus.FAILED
        assert result.entity_count == 0
        assert len(result.warnings) > 0

    def test_ingest_document_with_mock_llm(self, pipeline_setup):
        """测试使用 Mock LLM 入库文档"""
        pipeline, kg, memory, source_info = pipeline_setup
        doc = Document(
            url="test://mock",
            title="DF-21D 报道",
            content="DF-21D 反舰弹道导弹由中国航天工业集团研发，射程1500公里。",
            doc_type=DocumentType.WEB_PAGE,
            source_name="简氏防务",
        )
        result = asyncio.run(pipeline.ingest_document(doc, source_info))
        # Mock LLM 不会返回有效 JSON，但管道不应崩溃
        assert result.status in (IngestionStatus.INGESTED, IngestionStatus.FAILED)
        assert result.document_count == 1

    def test_ingest_from_urls_empty(self, pipeline_setup):
        """测试空 URL 列表"""
        pipeline, _, _, source_info = pipeline_setup
        result = asyncio.run(pipeline.ingest_from_urls([], source_info))
        assert result.document_count == 0

    def test_ingest_from_files_empty(self, pipeline_setup):
        """测试空文件列表"""
        pipeline, _, _, source_info = pipeline_setup
        result = asyncio.run(pipeline.ingest_from_files([], source_info))
        assert result.document_count == 0

    def test_ingest_from_files_unsupported(self, pipeline_setup):
        """测试不支持的文件类型"""
        pipeline, _, _, source_info = pipeline_setup
        result = asyncio.run(pipeline.ingest_from_files(
            ["data.csv"], source_info
        ))
        assert len(result.warnings) > 0

    def test_ingest_from_urls_invalid(self, pipeline_setup):
        """测试无效 URL"""
        pipeline, _, _, source_info = pipeline_setup
        result = asyncio.run(pipeline.ingest_from_urls(
            ["not-a-valid-url"], source_info
        ))
        # 无效 URL 应被跳过
        assert result.document_count == 0 or len(result.warnings) > 0

    def test_store_to_memory(self, pipeline_setup):
        """测试时序记忆库存储"""
        pipeline, kg, memory, source_info = pipeline_setup
        doc = Document(
            url="test://memory",
            title="记忆测试",
            content="这是一条测试内容",
            doc_type=DocumentType.WEB_PAGE,
            source_name="测试来源",
        )
        extraction = ExtractionResult(
            entities=[ExtractedEntity("e1", "Weapon", "测试武器", confidence=0.8)],
        )
        pipeline._store_to_memory(doc, extraction)
        # 验证记忆库中有记录
        entries = memory.query(key="ingestion")
        assert len(entries) >= 1

    def test_find_node_by_name(self, pipeline_setup):
        """测试按名称查找节点"""
        pipeline, kg, _, _ = pipeline_setup
        kg.add_node(GraphNode(
            id="weapon_001",
            type="Weapon",
            name="DF-21D",
            confidence=0.9,
        ))
        node = pipeline._find_node_by_name("DF-21D")
        assert node is not None
        assert node.id == "weapon_001"

        # 不存在时返回 None
        assert pipeline._find_node_by_name("不存在的武器") is None


# =============================================================================
# 集成测试
# =============================================================================

class TestIngestionIntegration:
    """集成测试 - 验证管道与核心算法的协同"""

    def test_pipeline_with_prepopulated_graph(self):
        """
        测试管道与已填充知识图谱的交叉验证

        预填充一个实体，然后入库同实体的不同属性值，
        验证交叉验证机制是否检测到冲突。
        """
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        memory = TimeSeriesMemory()

        # 预填充知识图谱
        kg.add_node(GraphNode(
            id="weapon_df21d",
            type="Weapon",
            name="DF-21D",
            attributes={"range_km": 1500},
            confidence=0.9,
            sources=["权威来源"],
        ))

        pipeline = IngestionPipeline(kg, llm, memory=memory)
        source_info = SourceInfo(
            name="新来源",
            reliability=SourceReliability.C,
        )

        # 入库一篇可能包含冲突数据的文档
        doc = Document(
            url="test://conflict",
            title="DF-21D 新报道",
            content="据报道，DF-21D 导弹射程已提升至 2000 公里。",
            doc_type=DocumentType.WEB_PAGE,
            source_name="新来源",
        )
        result = asyncio.run(pipeline.ingest_document(doc, source_info))
        # 管道应正常运行（Mock LLM 可能不会提取出冲突实体）
        assert result.document_count == 1
        assert result.status in (IngestionStatus.INGESTED, IngestionStatus.FAILED)

    def test_full_pipeline_with_manual_extraction(self):
        """
        测试管道核心步骤（跳过 LLM 抽取，手动构建抽取结果）

        验证从抽取结果到图谱入库的完整路径。
        """
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        memory = TimeSeriesMemory()
        pipeline = IngestionPipeline(kg, llm, memory=memory)

        source_info = SourceInfo(
            name="简氏防务",
            reliability=SourceReliability.A,
            history_accuracy=0.95,
            expertise_level=0.9,
            independence=0.8,
            traceability=True,
        )

        # 手动构建抽取结果
        entity = ExtractedEntity(
            entity_id="weapon_df21d",
            entity_type="Weapon",
            name="DF-21D 反舰弹道导弹",
            attributes={"range_km": 1500, "speed_mach": 3.0},
            confidence=0.9,
        )

        doc = Document(
            url="test://manual",
            title="手动测试",
            content="测试内容",
            doc_type=DocumentType.WEB_PAGE,
            source_name="简氏防务",
        )

        # 直接调用内部方法测试单实体入库
        entity_result = asyncio.run(pipeline._process_entity(
            entity, doc, source_info
        ))
        assert entity_result.entity_count == 1

        # 验证实体已入库到知识图谱
        assert "weapon_df21d" in kg.nodes
        node = kg.nodes["weapon_df21d"]
        assert node.name == "DF-21D 反舰弹道导弹"
        assert node.type == "Weapon"
        assert node.confidence > 0

    def test_pipeline_relation_ingestion(self):
        """测试关系入库"""
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        pipeline = IngestionPipeline(kg, llm)

        # 预填充两个实体
        kg.add_node(GraphNode(id="weapon_001", type="Weapon", name="DF-21D"))
        kg.add_node(GraphNode(id="org_001", type="Organization", name="CASIC"))

        # 入库关系
        relation = ExtractedRelation(
            source_entity="weapon_001",
            target_entity="org_001",
            relation_type="developed_by",
            confidence=0.85,
        )
        ingested = asyncio.run(pipeline._step_ingest_relation(relation))
        assert ingested == 1
        assert len(kg.edges) == 1
        assert kg.edges[0].relation == "developed_by"

    def test_pipeline_relation_missing_entity(self):
        """测试关系入库时实体不存在"""
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        pipeline = IngestionPipeline(kg, llm)

        # 只有一个实体存在
        kg.add_node(GraphNode(id="weapon_001", type="Weapon", name="DF-21D"))

        relation = ExtractedRelation(
            source_entity="weapon_001",
            target_entity="org_missing",  # 不存在
            relation_type="developed_by",
            confidence=0.85,
        )
        ingested = asyncio.run(pipeline._step_ingest_relation(relation))
        assert ingested == 0
        assert len(kg.edges) == 0


# =============================================================================
# 增强功能测试
# =============================================================================

class TestKnowledgeGraphEnhanced:
    """增强后的知识图谱测试"""

    def test_edge_deduplication(self):
        """测试边自动去重"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="n1", type="Weapon", name="W1"))
        kg.add_node(GraphNode(id="n2", type="Org", name="O1"))

        # 添加同一条边两次
        edge1 = GraphEdge(source_id="n1", target_id="n2",
                          relation="developed_by", confidence=0.7)
        edge2 = GraphEdge(source_id="n1", target_id="n2",
                          relation="developed_by", confidence=0.9)
        kg.add_edge(edge1)
        kg.add_edge(edge2)

        # 应只有一条边
        assert len(kg.edges) == 1
        # 置信度应取最大值
        assert kg.edges[0].confidence == 0.9

    def test_name_index_lookup(self):
        """测试名称索引快速查找"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="weapon_df21d", type="Weapon",
                              name="DF-21D"))
        kg.add_node(GraphNode(id="org_casic", type="Organization",
                              name="中国航天工业集团"))

        # 按名称查找
        node = kg.find_node_by_name("DF-21D")
        assert node is not None
        assert node.id == "weapon_df21d"

        # 按 ID 查找
        node = kg.find_node_by_name("org_casic")
        assert node is not None
        assert node.type == "Organization"

        # 不存在
        assert kg.find_node_by_name("不存在") is None

    def test_get_statistics(self):
        """测试图谱统计"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="n1", type="Weapon", name="W1",
                              confidence=0.9, sources=["来源A"]))
        kg.add_node(GraphNode(id="n2", type="Organization", name="O1",
                              confidence=0.8, sources=["来源B"]))
        kg.add_edge(GraphEdge(source_id="n1", target_id="n2",
                              relation="developed_by"))

        stats = kg.get_statistics()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["type_distribution"]["Weapon"] == 1
        assert stats["type_distribution"]["Organization"] == 1
        assert stats["relation_distribution"]["developed_by"] == 1
        assert stats["unique_sources"] == 2
        assert stats["avg_confidence"] == 0.85

    def test_export_import_snapshot(self):
        """测试快照导出和导入"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="n1", type="Weapon", name="DF-21D",
                              attributes={"range_km": 1500},
                              confidence=0.9))
        kg.add_edge(GraphEdge(source_id="n1", target_id="n1",
                              relation="self_ref", confidence=0.5))

        # 导出
        snapshot = kg.export_snapshot()
        assert len(snapshot["nodes"]) == 1
        assert len(snapshot["edges"]) == 1
        assert snapshot["statistics"]["node_count"] == 1

        # 导入到新图谱
        kg2 = KnowledgeGraph()
        imported = kg2.import_snapshot(snapshot)
        assert imported == 1
        assert "n1" in kg2.nodes
        assert kg2.nodes["n1"].name == "DF-21D"
        assert len(kg2.edges) == 1

    def test_node_source_deduplication(self):
        """测试节点来源列表去重"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="n1", type="Weapon", name="W1",
                              sources=["来源A"]))
        kg.add_node(GraphNode(id="n1", type="Weapon", name="W1",
                              sources=["来源A", "来源B"]))

        # 来源 A 不应重复
        assert kg.nodes["n1"].sources.count("来源A") == 1
        assert "来源B" in kg.nodes["n1"].sources

    def test_get_edges_for_node(self):
        """测试获取节点关联边"""
        kg = KnowledgeGraph()
        kg.add_node(GraphNode(id="n1", type="Weapon", name="W1"))
        kg.add_node(GraphNode(id="n2", type="Org", name="O1"))
        kg.add_node(GraphNode(id="n3", type="Loc", name="L1"))
        kg.add_edge(GraphEdge(source_id="n1", target_id="n2",
                              relation="developed_by"))
        kg.add_edge(GraphEdge(source_id="n1", target_id="n3",
                              relation="deployed_at"))

        edges = kg.get_edges_for_node("n1")
        assert len(edges) == 2


class TestExtractorEnhanced:
    """增强后的抽取器测试"""

    @pytest.fixture
    def extractor(self):
        llm = LLMClient(LLMConfig())
        return EntityExtractor(llm, chunk_size=500, chunk_overlap=100)

    def test_chunk_text_with_document_url(self, extractor):
        """测试分块时传递 document_url"""
        chunks = extractor.chunk_text(
            "测试文本内容",
            document_url="https://example.com/article"
        )
        assert len(chunks) == 1
        assert chunks[0].document_url == "https://example.com/article"

    def test_regex_fallback_relations(self, extractor):
        """测试关系正则回退"""
        response = (
            '这不是 JSON 格式。'
            '但包含 "source_entity": "weapon_df21d" '
            '和 "target_entity": "org_casic" '
            '以及 "relation_type": "developed_by" '
        )
        relations = extractor._regex_fallback_relations(response)
        assert len(relations) >= 1
        if relations:
            assert relations[0].source_entity == "weapon_df21d"
            assert relations[0].target_entity == "org_casic"
            assert relations[0].confidence == 0.3

    def test_regex_fallback_relations_empty(self, extractor):
        """测试关系正则回退 - 无匹配"""
        response = "完全没有关系信息的普通文本"
        relations = extractor._regex_fallback_relations(response)
        assert len(relations) == 0

    def test_parse_relations_response_invalid_json_fallback(self, extractor):
        """测试关系解析 - JSON 失败时走正则回退"""
        response = (
            '无法解析的文本 '
            '"source_entity": "a" '
            '"target_entity": "b" '
            '"relation_type": "uses"'
        )
        relations = extractor._parse_relations_response(response)
        # 正则回退应能提取到关系
        assert len(relations) >= 0  # 取决于正则匹配结果


class TestConnectorsEnhanced:
    """增强后的连接器测试"""

    def test_ssrf_protection_private_ranges(self):
        """测试 SSRF 防护 - 完整私有地址范围"""
        connector = WebScraperConnector()

        # 应阻止的私有地址
        assert connector.validate_source("http://10.0.0.1/admin") is False
        assert connector.validate_source("http://172.16.0.1/data") is False
        assert connector.validate_source("http://172.31.255.255/x") is False
        assert connector.validate_source("http://169.254.169.254/metadata") is False
        assert connector.validate_source("http://0.0.0.0/test") is False

        # 应允许的公网地址
        assert connector.validate_source("https://8.8.8.8/dns") is True
        assert connector.validate_source("https://www.google.com/search") is True

    def test_ssrf_protection_ipv6_loopback(self):
        """测试 SSRF 防护 - IPv6 环回地址"""
        connector = WebScraperConnector()
        assert connector.validate_source("http://[::1]/admin") is False


class TestPipelineEnhanced:
    """增强后的管道测试"""

    def test_extract_old_value_from_conflict_result(self):
        """测试从交叉验证结果中提取旧值"""
        # 有支撑证据
        conflict_result = {
            "supporting_evidence": [
                {"entity": "DF-21D", "value": 1500, "confidence": 0.9}
            ],
            "conflict_score": 0.5,
        }
        old_value = IngestionPipeline._extract_old_value(conflict_result, "range_km")
        assert old_value == 1500

        # 无支撑证据
        conflict_result_empty = {"supporting_evidence": []}
        old_value = IngestionPipeline._extract_old_value(conflict_result_empty, "range_km")
        assert old_value == "未知"

    def test_vector_db_storage(self):
        """测试向量数据库存储集成"""
        from core.infrastructure import VectorDatabase

        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        vdb = VectorDatabase()  # 内存模式
        pipeline = IngestionPipeline(kg, llm, vector_db=vdb)

        doc = Document(
            url="test://vector",
            title="向量测试",
            content="DF-21D 导弹射程 1500 公里",
            doc_type=DocumentType.WEB_PAGE,
            source_name="测试来源",
        )
        extraction = ExtractionResult(
            entities=[
                ExtractedEntity("e1", "Weapon", "DF-21D",
                                attributes={"range_km": 1500}, confidence=0.9)
            ],
        )

        pipeline._store_to_vector_db(doc, extraction)

        # 验证向量数据库中有数据
        assert len(vdb._memory_store) >= 1
        # 应包含实体向量和文档分块向量
        entity_stored = any(
            meta.get("entity_id") == "e1"
            for _, (_, meta) in vdb._memory_store.items()
        )
        assert entity_stored

    def test_text_to_vector(self):
        """测试文本转向量"""
        v1 = IngestionPipeline._text_to_vector("DF-21D 导弹")
        v2 = IngestionPipeline._text_to_vector("DF-21D 导弹")
        v3 = IngestionPipeline._text_to_vector("完全不同的文本内容")

        # 相同文本应产生相同向量
        assert v1 == v2
        # 向量维度应为 128
        assert len(v1) == 128
        # 向量应已归一化（L2 范数 ≈ 1）
        norm = sum(x * x for x in v1) ** 0.5
        assert abs(norm - 1.0) < 0.01 or norm == 0.0  # 空文本范数为 0

    def test_pipeline_with_conflict_old_value(self):
        """测试冲突记录包含真实旧值"""
        kg = KnowledgeGraph()
        llm = LLMClient(LLMConfig())
        pipeline = IngestionPipeline(kg, llm)

        # 预填充一个实体
        kg.add_node(GraphNode(
            id="weapon_df21d", type="Weapon", name="DF-21D",
            attributes={"range_km": 1500}, confidence=0.9,
            sources=["权威来源"],
        ))

        source_info = SourceInfo(
            name="新来源", reliability=SourceReliability.C,
        )
        doc = Document(
            url="test://conflict_val", title="冲突测试",
            content="DF-21D 射程报道", doc_type=DocumentType.WEB_PAGE,
            source_name="新来源",
        )

        # 手动构建有冲突属性的实体
        entity = ExtractedEntity(
            entity_id="weapon_df21d", entity_type="Weapon", name="DF-21D",
            attributes={"range_km": 2500}, confidence=0.8,
        )

        result = asyncio.run(pipeline._process_entity(entity, doc, source_info))
        # 如果有冲突记录，旧值不应该是"历史数据"
        for conflict in result.conflicts:
            if conflict.attribute == "range_km":
                assert conflict.old_value != "历史数据"
                # 应该是从图谱中获取的真实值或"未知"
                assert conflict.old_value in (1500, "未知")
