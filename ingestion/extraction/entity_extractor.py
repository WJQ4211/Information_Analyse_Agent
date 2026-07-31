"""
情报分析智能体 - LLM 实体/关系抽取器
Intelligence Analysis Agent - Entity & Relation Extractor

使用 LLM 从非结构化文本中抽取结构化实体和关系。
支持文本分块处理长文档，JSON 解析失败时有正则回退机制。
"""

import json
import re
import asyncio
from typing import Optional

from core.infrastructure import LLMClient
from ingestion.models import (
    DocumentChunk,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from ingestion.extraction.prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_TYPE_DESCRIPTIONS,
    RELATION_EXTRACTION_SYSTEM_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    RELATION_TYPE_DESCRIPTIONS,
)


# 默认分块参数
_DEFAULT_CHUNK_SIZE = 1000      # 每块最大字符数
_DEFAULT_CHUNK_OVERLAP = 200    # 块间重叠字符数


class EntityExtractor:
    """
    LLM 驱动的实体/关系抽取器

    工作流程：
    1. 将长文本分块（保留上下文重叠）
    2. 对每个分块调用 LLM 抽取实体
    3. 基于已抽取的实体调用 LLM 抽取关系
    4. 合并去重后返回结果
    """

    def __init__(self, llm_client: LLMClient,
                 chunk_size: int = _DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP):
        """
        Args:
            llm_client: LLM 客户端实例
            chunk_size: 文本分块大小（字符数）
            chunk_overlap: 分块间重叠字符数
        """
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def extract(self, text: str, document_url: str = "") -> ExtractionResult:
        """
        完整的抽取流程：实体抽取 → 关系抽取 → 合并结果

        使用 asyncio.gather 并行处理多个分块，提升长文档处理效率。

        Args:
            text: 原始文本
            document_url: 文档来源 URL（用于追踪）

        Returns:
            ExtractionResult 包含实体和关系
        """
        if not text or not text.strip():
            return ExtractionResult(parse_success=True)

        # 分块
        chunks = self.chunk_text(text, document_url=document_url)

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []
        all_raw_responses: list[str] = []
        overall_parse_success = True

        # 并行处理所有分块的实体抽取
        entity_tasks = [
            self._extract_entities_from_chunk(chunk) for chunk in chunks
        ]
        entity_results = await asyncio.gather(*entity_tasks, return_exceptions=True)

        # 收集实体抽取结果
        chunk_entities: list[list[ExtractedEntity]] = []
        for i, entity_result in enumerate(entity_results):
            if isinstance(entity_result, Exception):
                overall_parse_success = False
                chunk_entities.append([])
                continue
            chunk_entities.append(entity_result.entities)
            all_entities.extend(entity_result.entities)
            all_raw_responses.append(entity_result.raw_response)
            if not entity_result.parse_success:
                overall_parse_success = False

        # 并行处理关系抽取（每个分块使用该分块抽取到的实体）
        relation_tasks = []
        for i, chunk in enumerate(chunks):
            entities_for_chunk = chunk_entities[i] if i < len(chunk_entities) else []
            if entities_for_chunk:
                relation_tasks.append(
                    self._extract_relations_from_chunk(chunk, entities_for_chunk)
                )

        if relation_tasks:
            relation_results = await asyncio.gather(
                *relation_tasks, return_exceptions=True
            )
            for rel_result in relation_results:
                if isinstance(rel_result, Exception):
                    overall_parse_success = False
                    continue
                all_relations.extend(rel_result.relations)
                all_raw_responses.append(rel_result.raw_response)
                if not rel_result.parse_success:
                    overall_parse_success = False

        # 去重
        all_entities = self._deduplicate_entities(all_entities)
        all_relations = self._deduplicate_relations(all_relations)

        return ExtractionResult(
            entities=all_entities,
            relations=all_relations,
            raw_response="\n---\n".join(all_raw_responses),
            parse_success=overall_parse_success,
        )

    async def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """仅抽取实体（不抽取关系）"""
        result = await self.extract(text)
        return result.entities

    async def extract_relations(self, text: str,
                                 entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        """基于已有实体列表抽取关系"""
        if not entities:
            return []

        chunks = self.chunk_text(text)
        all_relations: list[ExtractedRelation] = []

        for chunk in chunks:
            relation_result = await self._extract_relations_from_chunk(chunk, entities)
            all_relations.extend(relation_result.relations)

        return self._deduplicate_relations(all_relations)

    # =========================================================================
    # 文本分块
    # =========================================================================

    def chunk_text(self, text: str,
                   chunk_size: Optional[int] = None,
                   chunk_overlap: Optional[int] = None,
                   document_url: str = "") -> list[DocumentChunk]:
        """
        将长文本切分为重叠的分块

        Args:
            text: 原始文本
            chunk_size: 每块最大字符数（默认使用构造函数参数）
            chunk_overlap: 块间重叠字符数
            document_url: 所属文档的 URL（用于追踪来源）

        Returns:
            DocumentChunk 列表
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap

        if not text or len(text) <= size:
            return [DocumentChunk(
                document_url=document_url,
                content=text or "",
                chunk_index=0,
                start_pos=0,
                end_pos=len(text or ""),
            )]

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + size, len(text))
            chunk_content = text[start:end]

            chunks.append(DocumentChunk(
                document_url=document_url,
                content=chunk_content,
                chunk_index=index,
                start_pos=start,
                end_pos=end,
            ))

            # 已到达文本末尾，无需继续
            if end >= len(text):
                break

            # 下一个块的起始位置（保留重叠）
            start = end - overlap
            index += 1

        return chunks

    # =========================================================================
    # 内部抽取方法
    # =========================================================================

    async def _extract_entities_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        """从单个文本块中抽取实体"""
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_type_descriptions=ENTITY_TYPE_DESCRIPTIONS,
            text=chunk.content,
        )

        response = await self.llm_client.generate(
            prompt,
            system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
        )

        entities = self._parse_entities_response(response)

        return ExtractionResult(
            entities=entities,
            raw_response=response,
            parse_success=len(entities) > 0 or '"entities": []' in response,
            chunk_index=chunk.chunk_index,
        )

    async def _extract_relations_from_chunk(
        self, chunk: DocumentChunk, entities: list[ExtractedEntity]
    ) -> ExtractionResult:
        """从单个文本块中抽取关系"""
        # 构建实体摘要
        entities_summary = "\n".join(
            f"  - {e.entity_id} ({e.entity_type}): {e.name}"
            for e in entities
        )

        prompt = RELATION_EXTRACTION_PROMPT.format(
            relation_type_descriptions=RELATION_TYPE_DESCRIPTIONS,
            entities_summary=entities_summary,
            text=chunk.content,
        )

        response = await self.llm_client.generate(
            prompt,
            system_prompt=RELATION_EXTRACTION_SYSTEM_PROMPT,
        )

        relations = self._parse_relations_response(response)

        return ExtractionResult(
            relations=relations,
            raw_response=response,
            parse_success=len(relations) > 0 or '"relations": []' in response,
            chunk_index=chunk.chunk_index,
        )

    # =========================================================================
    # JSON 解析（带回退）
    # =========================================================================

    @staticmethod
    def _parse_entities_response(response: str) -> list[ExtractedEntity]:
        """
        解析 LLM 实体抽取响应

        优先尝试 JSON 解析，失败时使用正则回退。
        """
        # 尝试直接解析
        try:
            data = json.loads(response)
            return EntityExtractor._build_entities_from_dict(data)
        except (json.JSONDecodeError, TypeError):
            pass

        # 尝试提取 JSON 块（LLM 可能在 JSON 外包裹了 markdown 代码块）
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return EntityExtractor._build_entities_from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # 尝试提取第一个 {...} 块
        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                return EntityExtractor._build_entities_from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # 正则回退：提取粗略实体信息
        return EntityExtractor._regex_fallback_entities(response)

    @staticmethod
    def _parse_relations_response(response: str) -> list[ExtractedRelation]:
        """
        解析 LLM 关系抽取响应

        优先尝试 JSON 解析，失败时使用正则回退。
        """
        try:
            data = json.loads(response)
            return EntityExtractor._build_relations_from_dict(data)
        except (json.JSONDecodeError, TypeError):
            pass

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return EntityExtractor._build_relations_from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass

        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                return EntityExtractor._build_relations_from_dict(data)
            except (json.JSONDecodeError, TypeError):
                pass

        # 正则回退：提取粗略关系信息
        return EntityExtractor._regex_fallback_relations(response)

    # =========================================================================
    # 构建数据结构
    # =========================================================================

    @staticmethod
    def _build_entities_from_dict(data: dict) -> list[ExtractedEntity]:
        """从解析后的字典构建实体列表"""
        entities = []
        for item in data.get("entities", []):
            try:
                entity = ExtractedEntity(
                    entity_id=str(item.get("entity_id", f"unknown_{len(entities)}")),
                    entity_type=str(item.get("entity_type", "Unknown")),
                    name=str(item.get("name", "未知")),
                    attributes=item.get("attributes", {}),
                    confidence=float(item.get("confidence", 0.5)),
                )
                # 校验置信度范围
                entity.confidence = max(0.0, min(1.0, entity.confidence))
                entities.append(entity)
            except (ValueError, TypeError):
                continue
        return entities

    @staticmethod
    def _build_relations_from_dict(data: dict) -> list[ExtractedRelation]:
        """从解析后的字典构建关系列表"""
        relations = []
        for item in data.get("relations", []):
            try:
                relation = ExtractedRelation(
                    source_entity=str(item.get("source_entity", "")),
                    target_entity=str(item.get("target_entity", "")),
                    relation_type=str(item.get("relation_type", "related_to")),
                    confidence=float(item.get("confidence", 0.5)),
                )
                # 校验置信度范围
                relation.confidence = max(0.0, min(1.0, relation.confidence))
                # 跳过空关系
                if relation.source_entity and relation.target_entity:
                    relations.append(relation)
            except (ValueError, TypeError):
                continue
        return relations

    @staticmethod
    def _regex_fallback_entities(response: str) -> list[ExtractedEntity]:
        """
        正则回退：当 JSON 解析完全失败时，尝试用正则提取粗略实体信息

        这是最后的兜底手段，抽取精度较低。
        """
        entities = []

        # 尝试匹配 "name": "xxx" 模式
        name_matches = re.findall(
            r'"(?:name|entity_name)"\s*:\s*"([^"]+)"', response
        )
        type_matches = re.findall(
            r'"(?:entity_type|type)"\s*:\s*"([^"]+)"', response
        )

        for i, name in enumerate(name_matches[:10]):  # 最多取 10 个
            entity_type = type_matches[i] if i < len(type_matches) else "Unknown"
            entity_id = f"fallback_{i}_{name[:20].replace(' ', '_')}"

            entities.append(ExtractedEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=name,
                confidence=0.3,  # 回退抽取，置信度低
            ))

        return entities

    @staticmethod
    def _regex_fallback_relations(response: str) -> list[ExtractedRelation]:
        """
        正则回退：当 JSON 解析完全失败时，尝试用正则提取粗略关系信息

        这是最后的兜底手段，抽取精度较低。
        """
        relations = []

        # 尝试匹配 source_entity / target_entity / relation_type 模式
        source_matches = re.findall(
            r'"(?:source_entity|source)"\s*:\s*"([^"]+)"', response
        )
        target_matches = re.findall(
            r'"(?:target_entity|target)"\s*:\s*"([^"]+)"', response
        )
        type_matches = re.findall(
            r'"(?:relation_type|relation|type)"\s*:\s*"([^"]+)"', response
        )

        # 取三者的最小长度，确保每个关系都有完整的三元组
        count = min(len(source_matches), len(target_matches), len(type_matches))
        for i in range(min(count, 10)):  # 最多取 10 条
            source = source_matches[i]
            target = target_matches[i]
            rel_type = type_matches[i]

            if source and target:
                relations.append(ExtractedRelation(
                    source_entity=source,
                    target_entity=target,
                    relation_type=rel_type,
                    confidence=0.3,  # 回退抽取，置信度低
                ))

        return relations

    # =========================================================================
    # 去重
    # =========================================================================

    @staticmethod
    def _deduplicate_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """
        实体去重

        按 entity_id 去重，保留置信度最高的版本。
        """
        seen: dict[str, ExtractedEntity] = {}
        for entity in entities:
            key = entity.entity_id
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())

    @staticmethod
    def _deduplicate_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
        """
        关系去重

        按 (source, target, relation_type) 三元组去重，保留置信度最高的版本。
        """
        seen: dict[tuple, ExtractedRelation] = {}
        for rel in relations:
            key = (rel.source_entity, rel.target_entity, rel.relation_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return list(seen.values())
