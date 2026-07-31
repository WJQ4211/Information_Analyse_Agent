"""
情报分析智能体 - 核心基础设施模块
Intelligence Analysis Agent - Core Infrastructure

包含：知识图谱、向量数据库、LLM客户端、时序记忆库
"""

import os
import json
import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# 知识图谱 (Knowledge Graph)
# =============================================================================

@dataclass
class GraphNode:
    """知识图谱节点"""
    id: str
    type: str  # Weapon, Organization, Person, Event, Location, Time
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    sources: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphEdge:
    """知识图谱边"""
    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)


class KnowledgeGraph:
    """
    知识图谱管理器

    使用 Neo4j 作为后端存储（可选），也支持内存模式用于开发和测试。
    支持：节点/边的增删查改、BFS 遍历、名称索引加速查找、
    图谱快照导出/导入、Neo4j 双向同步。
    """

    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, str] = {}  # name → node_id 快速查找
        self._edge_set: set[tuple[str, str, str]] = set()  # (src, tgt, rel) 去重

        # Neo4j 连接（可选）
        self._driver = None
        if neo4j_uri:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            except ImportError:
                pass  # 回退到内存模式

    def add_node(self, node: GraphNode) -> None:
        """添加节点到知识图谱"""
        if node.id in self.nodes:
            # 合并属性
            existing = self.nodes[node.id]
            existing.attributes.update(node.attributes)
            existing.confidence = max(existing.confidence, node.confidence)
            # 去重合并来源列表
            existing_sources_set = set(existing.sources)
            for src in node.sources:
                if src not in existing_sources_set:
                    existing.sources.append(src)
                    existing_sources_set.add(src)
        else:
            self.nodes[node.id] = node

        # 更新名称索引
        self._name_index[node.name] = node.id

        # 同步到 Neo4j
        if self._driver:
            self._sync_node_to_neo4j(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """添加边到知识图谱（自动去重）"""
        edge_key = (edge.source_id, edge.target_id, edge.relation)
        if edge_key in self._edge_set:
            # 边已存在，更新置信度和证据
            for existing_edge in self.edges:
                if (existing_edge.source_id == edge.source_id
                        and existing_edge.target_id == edge.target_id
                        and existing_edge.relation == edge.relation):
                    existing_edge.confidence = max(existing_edge.confidence, edge.confidence)
                    # 合并证据（去重）
                    existing_evidence_set = set(existing_edge.evidence)
                    for ev in edge.evidence:
                        if ev not in existing_evidence_set:
                            existing_edge.evidence.append(ev)
                            existing_evidence_set.add(ev)
                    break
            return

        self.edges.append(edge)
        self._edge_set.add(edge_key)
        self._adjacency[edge.source_id].append(edge.target_id)

        if self._driver:
            self._sync_edge_to_neo4j(edge)

    def query(self, entity: str = None, attribute: str = None,
              entity_type: str = None, time_window: tuple = None) -> list[GraphNode]:
        """
        查询知识图谱中的相关实体

        Args:
            entity: 实体名称或ID（可选）
            attribute: 属性名称（可选）
            entity_type: 实体类型（可选）
            time_window: 时间窗口 (start, end)（可选）

        Returns:
            匹配的节点列表
        """
        results = []

        for node_id, node in self.nodes.items():
            # 实体匹配
            if entity and entity not in [node.name, node.id]:
                continue

            # 类型匹配
            if entity_type and node.type != entity_type:
                continue

            # 属性匹配
            if attribute and attribute not in node.attributes:
                continue

            # 时间窗口匹配
            if time_window and 'timestamp' in node.attributes:
                ts = datetime.fromisoformat(node.attributes['timestamp'])
                if not (time_window[0] <= ts <= time_window[1]):
                    continue

            results.append(node)

        return results

    def get_related_nodes(self, node_id: str, max_depth: int = 3) -> list[GraphNode]:
        """获取与指定节点相关的节点（BFS）"""
        visited = set()
        queue = [(node_id, 0)]
        results = []

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            if current_id in self.nodes:
                results.append(self.nodes[current_id])

            for neighbor_id in self._adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))

        return results

    def get_entity_facts(self, entity_name: str = None, attribute: str = None) -> list[dict]:
        """获取实体的相关事实"""
        nodes = self.query(entity=entity_name) if entity_name else list(self.nodes.values())
        facts = []

        for node in nodes:
            for attr, value in node.attributes.items():
                if attribute and attr != attribute:
                    continue
                facts.append({
                    'entity': node.name,
                    'entity_type': node.type,
                    'attribute': attr,
                    'value': value,
                    'confidence': node.confidence,
                    'sources': node.sources,
                    'timestamp': node.created_at
                })

        return facts

    def find_node_by_name(self, name: str) -> Optional[GraphNode]:
        """
        按名称快速查找节点（使用名称索引，O(1) 复杂度）

        Args:
            name: 实体名称或实体 ID

        Returns:
            匹配的 GraphNode，未找到返回 None
        """
        # 先查名称索引
        node_id = self._name_index.get(name)
        if node_id and node_id in self.nodes:
            return self.nodes[node_id]

        # 回退：直接按 ID 查找
        if name in self.nodes:
            return self.nodes[name]

        return None

    def get_edges_for_node(self, node_id: str) -> list[GraphEdge]:
        """获取与指定节点关联的所有边"""
        return [e for e in self.edges
                if e.source_id == node_id or e.target_id == node_id]

    def get_statistics(self) -> dict:
        """
        获取知识图谱统计信息

        Returns:
            包含节点数、边数、类型分布等的字典
        """
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node.type] += 1

        relation_counts = defaultdict(int)
        for edge in self.edges:
            relation_counts[edge.relation] += 1

        avg_confidence = (
            sum(n.confidence for n in self.nodes.values()) / len(self.nodes)
            if self.nodes else 0.0
        )

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "type_distribution": dict(type_counts),
            "relation_distribution": dict(relation_counts),
            "avg_confidence": round(avg_confidence, 3),
            "unique_sources": len(set(
                src for node in self.nodes.values() for src in node.sources
            )),
        }

    def export_snapshot(self) -> dict:
        """
        导出知识图谱快照（JSON 可序列化）

        用于保存/传输知识图谱状态。

        Returns:
            包含所有节点和边的字典
        """
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "name": n.name,
                    "attributes": n.attributes,
                    "confidence": n.confidence,
                    "sources": n.sources,
                    "created_at": n.created_at,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "confidence": e.confidence,
                    "evidence": e.evidence,
                }
                for e in self.edges
            ],
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat(),
        }

    def import_snapshot(self, snapshot: dict) -> int:
        """
        从快照导入知识图谱数据

        Args:
            snapshot: export_snapshot() 生成的字典

        Returns:
            导入的节点数量
        """
        imported = 0
        for node_data in snapshot.get("nodes", []):
            node = GraphNode(
                id=node_data["id"],
                type=node_data["type"],
                name=node_data["name"],
                attributes=node_data.get("attributes", {}),
                confidence=node_data.get("confidence", 1.0),
                sources=node_data.get("sources", []),
                created_at=node_data.get("created_at", datetime.now().isoformat()),
            )
            self.add_node(node)
            imported += 1

        for edge_data in snapshot.get("edges", []):
            edge = GraphEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relation=edge_data["relation"],
                confidence=edge_data.get("confidence", 1.0),
                evidence=edge_data.get("evidence", []),
            )
            self.add_edge(edge)

        return imported

    def load_from_neo4j(self) -> int:
        """
        从 Neo4j 加载知识图谱数据到内存

        Returns:
            加载的节点数量，如果 Neo4j 未连接返回 0
        """
        if not self._driver:
            return 0

        loaded = 0
        try:
            with self._driver.session() as session:
                # 加载所有节点
                result = session.run(
                    "MATCH (n) RETURN n.id AS id, n.name AS name, "
                    "n.type AS type, n.confidence AS confidence, "
                    "n.attributes AS attributes, n.sources AS sources"
                )
                for record in result:
                    attrs = json.loads(record["attributes"]) if record["attributes"] else {}
                    sources = json.loads(record["sources"]) if record["sources"] else []
                    node = GraphNode(
                        id=record["id"],
                        type=record["type"] or "Unknown",
                        name=record["name"] or record["id"],
                        attributes=attrs,
                        confidence=record["confidence"] or 1.0,
                        sources=sources,
                    )
                    # 直接设置以避免重复同步
                    self.nodes[node.id] = node
                    self._name_index[node.name] = node.id
                    loaded += 1

                # 加载所有边
                result = session.run(
                    "MATCH (a)-[r:RELATION]->(b) "
                    "RETURN a.id AS source_id, b.id AS target_id, "
                    "r.relation AS relation, r.confidence AS confidence, "
                    "r.evidence AS evidence"
                )
                for record in result:
                    edge_key = (record["source_id"], record["target_id"], record["relation"])
                    if edge_key not in self._edge_set:
                        evidence = json.loads(record["evidence"]) if record["evidence"] else []
                        edge = GraphEdge(
                            source_id=record["source_id"],
                            target_id=record["target_id"],
                            relation=record["relation"],
                            confidence=record["confidence"] or 1.0,
                            evidence=evidence,
                        )
                        self.edges.append(edge)
                        self._edge_set.add(edge_key)
                        self._adjacency[edge.source_id].append(edge.target_id)

        except Exception:
            pass  # Neo4j 加载失败时静默处理

        return loaded

    def _sync_node_to_neo4j(self, node: GraphNode) -> None:
        """同步节点到 Neo4j"""
        if not self._driver:
            return

        with self._driver.session() as session:
            session.run(
                "MERGE (n {id: $id}) "
                "SET n += $props",
                id=node.id,
                props={
                    'name': node.name,
                    'type': node.type,
                    'confidence': node.confidence,
                    'attributes': json.dumps(node.attributes),
                    'sources': json.dumps(node.sources)
                }
            )

    def _sync_edge_to_neo4j(self, edge: GraphEdge) -> None:
        """同步边到 Neo4j"""
        if not self._driver:
            return

        with self._driver.session() as session:
            session.run(
                "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                "MERGE (a)-[r:RELATION {relation: $relation}]->(b) "
                "SET r.confidence = $confidence, r.evidence = $evidence",
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                confidence=edge.confidence,
                evidence=json.dumps(edge.evidence)
            )

    def close(self) -> None:
        """关闭连接"""
        if self._driver:
            self._driver.close()


# =============================================================================
# 向量数据库 (Vector Database)
# =============================================================================

class VectorDatabase:
    """
    向量数据库管理器

    使用 Qdrant 或 Milvus 作为后端存储，也支持内存模式。
    """

    def __init__(self, host: str = None, port: int = None):
        self._client = None
        self._memory_store: dict[str, tuple[list[float], dict]] = {}

        if host:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(host=host, port=port)
            except ImportError:
                pass

    def store(self, vector_id: str, vector: list[float], metadata: dict) -> None:
        """存储向量"""
        if self._client:
            self._client.upsert(
                collection_name="intelligence",
                points=[(vector_id, vector, metadata)]
            )
        else:
            self._memory_store[vector_id] = (vector, metadata)

    def search(self, query_vector: list[float], top_k: int = 10,
               score_threshold: float = 0.7) -> list[tuple[str, float, dict]]:
        """向量相似度搜索"""
        results = []

        if self._client:
            hits = self._client.search(
                collection_name="intelligence",
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            for hit in hits:
                results.append((hit.id, hit.score, hit.payload))
        else:
            # 内存模式：余弦相似度
            import math

            def cosine_similarity(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(y * y for y in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot / (norm_a * norm_b)

            for vid, (vec, meta) in self._memory_store.items():
                score = cosine_similarity(query_vector, vec)
                if score >= score_threshold:
                    results.append((vid, score, meta))

            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        return results


# =============================================================================
# LLM 客户端 (LLM Client)
# =============================================================================

@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = "qwen2-72b"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "local"
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.9


class LLMClient:
    """
    LLM 客户端

    支持本地部署的 Llama-3-70B 或 Qwen-2-72B。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=config.api_base,
                api_key=config.api_key
            )
        except ImportError:
            pass

    async def generate(self, prompt: str, system_prompt: str = None,
                       temperature: float = None) -> str:
        """异步生成文本"""
        if not self._client:
            return self._mock_generate(prompt, system_prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )

        return response.choices[0].message.content

    def generate_sync(self, prompt: str, system_prompt: str = None,
                      temperature: float = None) -> str:
        """同步生成文本"""
        if not self._client:
            return self._mock_generate(prompt, system_prompt)
        try:
            loop = asyncio.get_running_loop()
            # 在运行的事件循环中，使用线程池执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate(prompt, system_prompt, temperature)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.generate(prompt, system_prompt, temperature))

    def _mock_generate(self, prompt: str, system_prompt: str = None) -> str:
        """模拟生成（用于开发测试）"""
        return f"[Mock LLM] 基于提示生成的响应：{prompt[:100]}..."


# =============================================================================
# 时序记忆库 (Time-Series Memory)
# =============================================================================

@dataclass
class MemoryEntry:
    """记忆条目"""
    timestamp: datetime
    content: str
    metadata: dict
    embedding: list[float] = None


class TimeSeriesMemory:
    """
    时序记忆库

    用于存储情报分析过程中的时序信息，支持弱信号检测和趋势分析。
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries
        self._index: dict[str, list[int]] = defaultdict(list)

    def add(self, content: str, metadata: dict, timestamp: datetime = None,
            embedding: list[float] = None) -> None:
        """添加记忆条目"""
        entry = MemoryEntry(
            timestamp=timestamp or datetime.now(),
            content=content,
            metadata=metadata,
            embedding=embedding
        )

        self._entries.append(entry)

        # 建立索引
        for key, value in metadata.items():
            self._index[f"{key}:{value}"].append(len(self._entries) - 1)

        # 限制容量
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def query(self, key: str = None, value: str = None,
              time_range: tuple = None) -> list[MemoryEntry]:
        """查询记忆"""
        if key and value:
            indices = self._index.get(f"{key}:{value}", [])
            results = [self._entries[i] for i in indices]
        else:
            results = list(self._entries)

        # 时间过滤
        if time_range:
            results = [
                e for e in results
                if time_range[0] <= e.timestamp <= time_range[1]
            ]

        return results

    def get_timeline(self, key: str = None, value: str = None,
                     interval: str = "daily") -> list[dict]:
        """获取时间线数据"""
        entries = self.query(key, value)

        # 按时间间隔聚合
        timeline = defaultdict(list)

        for entry in entries:
            if interval == "daily":
                bucket = entry.timestamp.date()
            elif interval == "weekly":
                bucket = entry.timestamp.isocalendar()[:2]
            elif interval == "monthly":
                bucket = (entry.timestamp.year, entry.timestamp.month)
            else:
                bucket = entry.timestamp

            timeline[bucket].append(entry)

        # 聚合结果
        result = []
        for bucket, bucket_entries in sorted(timeline.items()):
            result.append({
                'time': str(bucket),
                'count': len(bucket_entries),
                'entries': [e.content for e in bucket_entries],
                'avg_confidence': sum(e.metadata.get('confidence', 0.5) for e in bucket_entries) / len(bucket_entries)
            })

        return result
