"""
情报分析智能体 - 核心算法模块
Intelligence Analysis Agent - Core Algorithms

包含：
1. NATO 6×6 情报评估算法
2. GraphRAG 跨验证算法
3. DOTMLPF-P 薄弱点推演算法
4. 供应链异常检测算法
5. ACH 竞争性假说分析算法
6. TRL 轨迹预测算法
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import defaultdict

from core.infrastructure import KnowledgeGraph, LLMClient, TimeSeriesMemory


# =============================================================================
# 1. NATO 6×6 情报评估算法
# =============================================================================

class SourceReliability(Enum):
    """信息源可靠度等级 (A-F)"""
    A = "A"  # 极其可靠
    B = "B"  # 相当可靠
    C = "C"  # 一般可靠
    D = "D"  # 可疑
    E = "E"  # 不可靠
    F = "F"  # 无法评定


class InfoCredibility(Enum):
    """信息可信度等级 (1-6)"""
    ONE = 1    # 无法确认
    TWO = 2    # 极不可信
    THREE = 3  # 不可信
    FOUR = 4   # 一般可信
    FIVE = 5   # 相当可信
    SIX = 6    # 极其可信


@dataclass
class SourceInfo:
    """信息源信息"""
    name: str
    reliability: SourceReliability
    history_accuracy: float = 0.5  # [0, 1]
    expertise_level: float = 0.5   # [0, 1]
    independence: float = 0.5      # [0, 1]
    traceability: bool = False


@dataclass
class Evidence:
    """证据条目"""
    content: str
    source: SourceInfo
    timestamp: datetime
    entity: str = None
    attribute: str = None
    value: Any = None
    cross_references: list[str] = field(default_factory=list)


@dataclass
class AssessmentResult:
    """评估结果"""
    source_reliability: SourceReliability
    info_credibility: InfoCredibility
    combined_confidence: float  # [0, 1]
    confidence_level: str  # "high", "medium", "low"
    maskirovka_flag: bool = False
    supporting_sources: list[str] = field(default_factory=list)


# 可靠度等级到数值的映射
SR_NORM = {
    SourceReliability.A: 1.0,
    SourceReliability.B: 0.85,
    SourceReliability.C: 0.70,
    SourceReliability.D: 0.55,
    SourceReliability.E: 0.40,
    SourceReliability.F: 0.25,
}

# 可信度等级到数值的映射
IC_NORM = {
    InfoCredibility.SIX: 1.0,
    InfoCredibility.FIVE: 0.85,
    InfoCredibility.FOUR: 0.70,
    InfoCredibility.THREE: 0.55,
    InfoCredibility.TWO: 0.40,
    InfoCredibility.ONE: 0.25,
}


def calculate_source_reliability(source: SourceInfo) -> SourceReliability:
    """
    计算信息源可靠度 (NATO 6×6 标准)

    SR_score = 0.4 * history_accuracy + 0.3 * expertise_level
               + 0.2 * independence + 0.1 * traceability
    """
    traceability_score = 1.0 if source.traceability else 0.0
    sr_score = (
        0.4 * source.history_accuracy
        + 0.3 * source.expertise_level
        + 0.2 * source.independence
        + 0.1 * traceability_score
    )

    # 映射到等级
    if sr_score >= 0.85:
        return SourceReliability.A
    elif sr_score >= 0.70:
        return SourceReliability.B
    elif sr_score >= 0.55:
        return SourceReliability.C
    elif sr_score >= 0.40:
        return SourceReliability.D
    elif sr_score >= 0.25:
        return SourceReliability.E
    else:
        return SourceReliability.F


def calculate_info_credibility(evidence: Evidence,
                               knowledge_graph: KnowledgeGraph = None) -> InfoCredibility:
    """
    计算信息可信度 (NATO 6×6 标准)

    基于交叉印证情况和来源可靠度。
    """
    cross_ref_count = len(evidence.cross_references)

    # 交叉印证来源的可靠度
    if knowledge_graph and evidence.cross_references:
        ref_reliabilities = []
        for ref_id in evidence.cross_references:
            ref_facts = knowledge_graph.get_entity_facts(evidence.entity, evidence.attribute)
            for fact in ref_facts:
                for src in fact.get('sources', []):
                    # 查找来源信息
                    pass
        avg_reliability = 0.5  # 默认
    else:
        avg_reliability = SR_NORM.get(evidence.source.reliability, 0.5)

    # 一致性评分（简化：如果有交叉印证，假设一致性较高）
    consistency = min(1.0, cross_ref_count / 3.0) if cross_ref_count > 0 else 0.0

    # 归一化交叉印证数量
    normalized_count = min(1.0, cross_ref_count / 3.0)

    ic_score = (
        0.3 * normalized_count
        + 0.4 * avg_reliability
        + 0.3 * consistency
    )

    # 映射到等级
    if ic_score >= 0.85:
        return InfoCredibility.SIX
    elif ic_score >= 0.70:
        return InfoCredibility.FIVE
    elif ic_score >= 0.55:
        return InfoCredibility.FOUR
    elif ic_score >= 0.40:
        return InfoCredibility.THREE
    elif ic_score >= 0.25:
        return InfoCredibility.TWO
    else:
        return InfoCredibility.ONE


def assess_evidence(evidence: Evidence,
                    knowledge_graph: KnowledgeGraph = None) -> AssessmentResult:
    """
    对证据进行完整的 NATO 6×6 评估

    返回包含可靠度、可信度、综合置信度的评估结果。
    """
    # 1. 计算源可靠度
    sr = calculate_source_reliability(evidence.source)

    # 2. 计算信息可信度
    ic = calculate_info_credibility(evidence, knowledge_graph)

    # 3. 计算综合置信度
    sr_weight = 0.4
    ic_weight = 0.6
    combined_confidence = (
        sr_weight * SR_NORM[sr]
        + ic_weight * IC_NORM[ic]
    )

    # 4. 分类置信度等级
    if combined_confidence >= 0.75:
        confidence_level = "high"
    elif combined_confidence >= 0.50:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # 5. 检测战略欺骗 (Maskirovka)
    maskirovka_flag = False
    if knowledge_graph and evidence.entity and evidence.attribute:
        historical_facts = knowledge_graph.get_entity_facts(evidence.entity, evidence.attribute)
        for fact in historical_facts:
            historical_value = fact.get('value')
            if historical_value and evidence.value:
                try:
                    # 数值冲突检测
                    if isinstance(historical_value, (int, float)) and isinstance(evidence.value, (int, float)):
                        conflict_ratio = abs(evidence.value - historical_value) / max(abs(historical_value), 1)
                        if conflict_ratio > 0.3 and sr in [SourceReliability.D, SourceReliability.E, SourceReliability.F]:
                            maskirovka_flag = True
                            break
                except (TypeError, ZeroDivisionError):
                    pass

    return AssessmentResult(
        source_reliability=sr,
        info_credibility=ic,
        combined_confidence=combined_confidence,
        confidence_level=confidence_level,
        maskirovka_flag=maskirovka_flag,
        supporting_sources=evidence.cross_references
    )


# =============================================================================
# 2. GraphRAG 跨验证算法
# =============================================================================

def cross_validate_fact(entity: str, attribute: str, value: Any,
                        knowledge_graph: KnowledgeGraph,
                        time_window: timedelta = None) -> dict:
    """
    GraphRAG 跨验证算法

    在知识图谱中查找与新事实相关的历史事实，
    计算冲突分数并检测战略欺骗。
    """
    # 1. 查询相关历史事实
    if time_window:
        now = datetime.now()
        window_start = now - time_window
        related_facts = knowledge_graph.query(
            entity=entity,
            attribute=attribute,
            time_window=(window_start, now)
        )
    else:
        related_facts = knowledge_graph.query(entity=entity, attribute=attribute)

    # 2. 计算冲突分数
    if not related_facts:
        return {
            "is_confirmed": False,
            "conflict_score": 0.0,
            "maskirovka_flag": False,
            "supporting_evidence": [],
            "message": "无历史事实可供交叉验证"
        }

    conflict_scores = []
    supporting_facts = []

    for fact_node in related_facts:
        historical_value = fact_node.attributes.get(attribute)
        if historical_value is None:
            continue

        try:
            if isinstance(historical_value, (int, float)) and isinstance(value, (int, float)):
                if historical_value == value:
                    conflict_scores.append(0.0)
                    supporting_facts.append(fact_node)
                else:
                    conflict_ratio = abs(value - historical_value) / max(abs(historical_value), 1)
                    conflict_scores.append(conflict_ratio)
            elif historical_value == value:
                conflict_scores.append(0.0)
                supporting_facts.append(fact_node)
            else:
                conflict_scores.append(1.0)  # 完全冲突
        except (TypeError, ZeroDivisionError):
            conflict_scores.append(0.5)

    avg_conflict = sum(conflict_scores) / len(conflict_scores) if conflict_scores else 0.0
    max_conflict = max(conflict_scores) if conflict_scores else 0.0

    # 3. 确定验证结果
    is_confirmed = avg_conflict < 0.1
    maskirovka_flag = (max_conflict > 0.5 and avg_conflict > 0.3)

    return {
        "is_confirmed": is_confirmed,
        "conflict_score": avg_conflict,
        "max_conflict": max_conflict,
        "maskirovka_flag": maskirovka_flag,
        "supporting_evidence": [
            {
                "entity": f.name,
                "value": f.attributes.get(attribute),
                "confidence": f.confidence,
                "sources": f.sources
            }
            for f in supporting_facts
        ],
        "message": "事实已确认" if is_confirmed else "存在冲突，需要进一步验证" if avg_conflict > 0.1 else "无法确认"
    }


def update_knowledge_graph(graph: KnowledgeGraph, entity: str, entity_type: str,
                           attribute: str, value: Any, confidence: float,
                           source: str, timestamp: datetime = None) -> None:
    """
    更新知识图谱

    如果实体已存在，更新属性值；否则创建新实体。
    """
    from core.infrastructure import GraphNode, GraphEdge

    # 生成实体 ID
    entity_id = f"{entity}_{entity_type}"

    # 创建或更新节点
    node = GraphNode(
        id=entity_id,
        type=entity_type,
        name=entity,
        attributes={
            attribute: value,
            "timestamp": (timestamp or datetime.now()).isoformat()
        },
        confidence=confidence,
        sources=[source]
    )

    graph.add_node(node)

    # 添加溯源边
    source_node = GraphNode(
        id=f"source_{source}",
        type="Source",
        name=source,
        confidence=1.0
    )
    graph.add_node(source_node)

    evidence_edge = GraphEdge(
        source_id=entity_id,
        target_id=f"source_{source}",
        relation="evidenced_by",
        confidence=confidence,
        evidence=[source]
    )
    graph.add_edge(evidence_edge)


# =============================================================================
# 3. DOTMLPF-P 薄弱点推演算法
# =============================================================================

class DOTMLPFPDimension(Enum):
    """DOTMLPF-P 维度枚举"""
    DOCTRINE = "Doctrine"        # 条令
    ORGANIZATION = "Organization"  # 组织
    TRAINING = "Training"        # 训练
    MATERIEL = "Materiel"        # 物资
    LEADERSHIP = "Leadership"    # 领导力
    PERSONNEL = "Personnel"      # 人员
    FACILITIES = "Facilities"    # 设施
    POLICY = "Policy"            # 政策


# 维度权重
DIMENSION_WEIGHTS = {
    DOTMLPFPDimension.DOCTRINE: 0.12,
    DOTMLPFPDimension.ORGANIZATION: 0.10,
    DOTMLPFPDimension.TRAINING: 0.15,
    DOTMLPFPDimension.MATERIEL: 0.20,
    DOTMLPFPDimension.LEADERSHIP: 0.10,
    DOTMLPFPDimension.PERSONNEL: 0.10,
    DOTMLPFPDimension.FACILITIES: 0.15,
    DOTMLPFPDimension.POLICY: 0.08,
}


@dataclass
class DimensionEvidence:
    """维度证据"""
    content: str
    type: str  # "strength", "weakness", "neutral"
    confidence: float
    source: str
    timestamp: datetime


@dataclass
class Vulnerability:
    """薄弱点"""
    dimension: DOTMLPFPDimension
    weakness_score: float
    supporting_evidence: list[DimensionEvidence]
    confidence: float
    description: str = ""


def sigmoid(x: float) -> float:
    """Sigmoid 函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def score_dimension(dimension: DOTMLPFPDimension,
                    evidence_set: list[DimensionEvidence]) -> float:
    """
    计算 DOTMLPF-P 维度薄弱度评分

    薄弱度 = 1 - 强度
    强度通过 Sigmoid 函数归一化到 [0, 1]
    """
    if not evidence_set:
        return 1.0  # 无证据表明强项，假设存在薄弱

    strength_score = 0.0
    for evidence in evidence_set:
        if evidence.type == "strength":
            strength_score += evidence.confidence * 1.0
        elif evidence.type == "weakness":
            strength_score -= evidence.confidence * 1.0
        # neutral: 贡献为 0

    # 归一化
    normalized_strength = sigmoid(strength_score / max(1, len(evidence_set)))
    weakness_score = 1.0 - normalized_strength

    return weakness_score


def extract_dimension_evidence(knowledge_graph: KnowledgeGraph,
                               dimension: DOTMLPFPDimension) -> list[DimensionEvidence]:
    """
    从知识图谱中提取与指定维度相关的证据

    使用 LLM 进行维度分类和证据类型判断。
    """
    # 查询相关实体
    entities = knowledge_graph.query(entity_type=dimension.value)
    if not entities:
        # 尝试模糊匹配
        entities = knowledge_graph.query(entity=dimension.value)

    evidence_list = []
    for node in entities:
        for attr, value in node.attributes.items():
            evidence_list.append(DimensionEvidence(
                content=f"{node.name} - {attr}: {value}",
                type="neutral",  # 默认中性，实际应通过 LLM 判断
                confidence=node.confidence,
                source=node.sources[0] if node.sources else "unknown",
                timestamp=datetime.fromisoformat(node.created_at)
            ))

    return evidence_list


def deduce_vulnerabilities(knowledge_graph: KnowledgeGraph,
                           threshold: float = 0.5) -> list[Vulnerability]:
    """
    DOTMLPF-P 薄弱点推演算法

    对每个维度进行评分，识别超过阈值的薄弱点。
    """
    vulnerabilities = []

    for dimension in DOTMLPFPDimension:
        # 提取维度证据
        evidence_set = extract_dimension_evidence(knowledge_graph, dimension)

        # 计算薄弱度
        weakness_score = score_dimension(dimension, evidence_set)

        # 超过阈值则视为薄弱点
        if weakness_score >= threshold:
            # 计算置信度
            if evidence_set:
                confidence = sum(e.confidence for e in evidence_set) / len(evidence_set)
            else:
                confidence = 0.3  # 无证据时的低置信度

            # 生成描述
            description = generate_vulnerability_description(dimension, evidence_set, weakness_score)

            vulnerabilities.append(Vulnerability(
                dimension=dimension,
                weakness_score=weakness_score,
                supporting_evidence=evidence_set,
                confidence=confidence,
                description=description
            ))

    # 按薄弱度排序
    vulnerabilities.sort(key=lambda v: v.weakness_score, reverse=True)

    return vulnerabilities


def generate_vulnerability_description(dimension: DOTMLPFPDimension,
                                       evidence_set: list[DimensionEvidence],
                                       weakness_score: float) -> str:
    """生成薄弱点描述"""
    dimension_names = {
        DOTMLPFPDimension.DOCTRINE: "条令",
        DOTMLPFPDimension.ORGANIZATION: "组织",
        DOTMLPFPDimension.TRAINING: "训练",
        DOTMLPFPDimension.MATERIEL: "物资",
        DOTMLPFPDimension.LEADERSHIP: "领导力",
        DOTMLPFPDimension.PERSONNEL: "人员",
        DOTMLPFPDimension.FACILITIES: "设施",
        DOTMLPFPDimension.POLICY: "政策",
    }

    weakness_level = "严重" if weakness_score > 0.7 else "中等" if weakness_score > 0.5 else "轻微"

    evidence_summary = "; ".join([e.content for e in evidence_set[:3]]) if evidence_set else "缺乏相关证据"

    return f"敌方{dimension_names[dimension]}维存在{weakness_level}薄弱点（薄弱度: {weakness_score:.2f}）。依据证据: {evidence_summary}"


def compose_vulnerabilities(vulnerabilities: list[Vulnerability]) -> list[dict]:
    """
    组合推理：将多个维度的薄弱点组合成更高层次的薄弱点描述

    例如：物资薄弱 + 训练薄弱 → 后勤瓶颈
    """
    composite = []

    # 识别相关维度组
    related_groups = identify_related_dimensions(vulnerabilities)

    for group in related_groups:
        if len(group) >= 2:
            combined_score = sum(v.weakness_score for v in group) / len(group)
            if combined_score > 0.6:
                composite.append({
                    "type": "composite_vulnerability",
                    "dimensions": [v.dimension.value for v in group],
                    "score": combined_score,
                    "description": f"多个维度（{[v.dimension.value for v in group]}）存在组合薄弱，形成系统性风险",
                    "impact": assess_impact(group)
                })

    return composite


def identify_related_dimensions(vulnerabilities: list[Vulnerability]) -> list[list[Vulnerability]]:
    """识别相关维度组"""
    # 简单的分组策略：物资+训练+设施 → 后勲链；人员+领导力 → 人力链
    groups = []

    logistics_dims = [DOTMLPFPDimension.MATERIEL, DOTMLPFPDimension.TRAINING, DOTMLPFPDimension.FACILITIES]
    human_dims = [DOTMLPFPDimension.PERSONNEL, DOTMLPFPDimension.LEADERSHIP]

    logistics_vulns = [v for v in vulnerabilities if v.dimension in logistics_dims]
    human_vulns = [v for v in vulnerabilities if v.dimension in human_dims]

    if len(logistics_vulns) >= 2:
        groups.append(logistics_vulns)
    if len(human_vulns) >= 2:
        groups.append(human_vulns)

    # 单维度薄弱点
    for v in vulnerabilities:
        if v not in [item for group in groups for item in group]:
            groups.append([v])

    return groups


def assess_impact(vulnerabilities: list[Vulnerability]) -> str:
    """评估薄弱点的战术影响"""
    avg_score = sum(v.weakness_score for v in vulnerabilities) / len(vulnerabilities)
    if avg_score > 0.7:
        return "高影响：可能导致战斗力显著下降"
    elif avg_score > 0.5:
        return "中影响：可能影响特定作战能力"
    else:
        return "低影响：有限的战术影响"


# =============================================================================
# 4. 供应链异常检测算法
# =============================================================================

def detect_procurement_anomaly(procurement_history: list[dict],
                               window: int = 10) -> dict:
    """
    采购合同异常检测算法

    结合统计检测和 Isolation Forest 进行异常检测。
    """
    if len(procurement_history) < 3:
        return {"is_anomaly": False, "anomaly_score": 0.0, "type": "insufficient_data"}

    # 1. 计算基线
    amounts = [r.get('amount', 0) for r in procurement_history[:-1]]
    intervals = []
    for i in range(1, len(procurement_history)):
        t1 = datetime.fromisoformat(procurement_history[i-1].get('timestamp', '2000-01-01'))
        t2 = datetime.fromisoformat(procurement_history[i].get('timestamp', '2000-01-01'))
        intervals.append((t2 - t1).total_seconds())

    avg_amount = sum(amounts) / len(amounts) if amounts else 0
    std_amount = (sum((a - avg_amount) ** 2 for a in amounts) / len(amounts)) ** 0.5 if amounts else 1
    avg_interval = sum(intervals[:-1]) / len(intervals[:-1]) if len(intervals) > 1 else 1
    std_interval = (sum((i - avg_interval) ** 2 for i in intervals[:-1]) / len(intervals[:-1])) ** 0.5 if len(intervals) > 1 else 1

    # 2. 统计检测
    latest = procurement_history[-1]
    latest_amount = latest.get('amount', 0)
    latest_interval = intervals[-1] if intervals else 0

    z_score_amount = abs(latest_amount - avg_amount) / max(std_amount, 1) if std_amount > 0 else 0
    z_score_interval = abs(latest_interval - avg_interval) / max(std_interval, 1) if std_interval > 0 else 0

    if z_score_amount > 3.0 or z_score_interval > 3.0:
        return {
            "is_anomaly": True,
            "anomaly_score": min(1.0, max(z_score_amount, z_score_interval) / 5.0),
            "type": "statistical_outlier",
            "details": {
                "amount_z_score": z_score_amount,
                "interval_z_score": z_score_interval
            }
        }

    # 3. 语义异常检测
    common_categories = set()
    for r in procurement_history[:-1]:
        common_categories.add(r.get('category', 'unknown'))

    if latest.get('category') not in common_categories:
        return {
            "is_anomaly": True,
            "anomaly_score": 0.8,
            "type": "semantic_anomaly",
            "details": {"unexpected_category": latest.get('category')}
        }

    return {"is_anomaly": False, "anomaly_score": 0.0, "type": "normal"}


def detect_academic_anomaly(citation_network: dict) -> dict:
    """
    学术引用网络异常检测算法

    检测引用集中度异常、引用激增、基础研究缺失等信号。
    """
    papers = citation_network.get('papers', [])
    citations = citation_network.get('citations', [])

    if not papers:
        return {"is_anomaly": False, "anomaly_score": 0.0, "type": "no_data"}

    # 1. 计算 H 指数
    citation_counts = [p.get('citation_count', 0) for p in papers]
    h_index = 0
    sorted_counts = sorted(citation_counts, reverse=True)
    for i, count in enumerate(sorted_counts):
        if count >= i + 1:
            h_index = i + 1
        else:
            break

    # 2. 检测引用激增
    # 使用数据中的最大年份作为"当前"年份
    max_year = max([c.get('year', 0) for c in citations], default=datetime.now().year)
    recent_threshold = max_year - 1

    recent_citations = sum(
        c.get('count', 0) for c in citations
        if c.get('year', 0) >= recent_threshold
    )
    historical_count = len([c for c in citations if c.get('year', 0) < recent_threshold])
    historical_avg = sum(
        c.get('count', 0) for c in citations
        if c.get('year', 0) < recent_threshold
    ) / max(1, historical_count)

    if recent_citations > historical_avg * 3 and historical_avg > 0:
        return {
            "is_anomaly": True,
            "anomaly_score": min(1.0, recent_citations / (historical_avg * 3)),
            "type": "citation_spike",
            "details": {
                "recent_citations": recent_citations,
                "historical_avg": historical_avg,
                "h_index": h_index
            }
        }

    # 3. 检测基础研究缺失
    foundational_papers = [p for p in papers if p.get('type') == 'foundational']
    foundational_ratio = len(foundational_papers) / max(1, len(papers))

    if foundational_ratio < 0.1:
        return {
            "is_anomaly": True,
            "anomaly_score": 0.7,
            "type": "lack_of_fundamentals",
            "details": {
                "foundational_ratio": foundational_ratio,
                "h_index": h_index
            }
        }

    return {"is_anomaly": False, "anomaly_score": 0.0, "type": "normal"}


def detect_social_anomaly(social_media_posts: list[dict]) -> dict:
    """
    社交媒体情感异常检测算法

    检测负面情绪集中、关键词异常等信号。
    """
    if not social_media_posts:
        return {"is_anomaly": False, "anomaly_score": 0.0, "type": "no_data"}

    # 1. 情感分析（简化版）
    negative_keywords = ["缺少", "不足", "短缺", "故障", "问题", "失败", "推迟", "取消"]
    negative_count = 0
    total_count = len(social_media_posts)

    for post in social_media_posts:
        content = post.get('content', '')
        for kw in negative_keywords:
            if kw in content:
                negative_count += 1
                break

    negative_ratio = negative_count / total_count

    # 2. 异常判断
    if negative_ratio > 0.3:
        return {
            "is_anomaly": True,
            "anomaly_score": negative_ratio,
            "type": "social_unrest_signal",
            "details": {
                "negative_ratio": negative_ratio,
                "negative_count": negative_count,
                "total_count": total_count,
                "sample_evidence": [p.get('content', '')[:100] for p in social_media_posts if any(kw in p.get('content', '') for kw in negative_keywords)][:3]
            }
        }

    return {"is_anomaly": False, "anomaly_score": 0.0, "type": "normal"}


# =============================================================================
# 5. ACH 竞争性假说分析算法
# =============================================================================

@dataclass
class Hypothesis:
    """竞争性假说"""
    id: str
    description: str
    probability: float = 0.0


@dataclass
class EvidenceEvaluation:
    """证据对假说的评估"""
    evidence_id: str
    evaluation: str  # "support", "contradict", "neutral", "single_source"


def generate_hypotheses(topic: str, knowledge_graph: KnowledgeGraph,
                        llm_client: LLMClient = None) -> list[Hypothesis]:
    """
    生成互斥的竞争性假说

    使用 LLM 生成初始假说，并验证互斥性。
    """
    # 从知识图谱中提取相关事实
    facts = knowledge_graph.get_entity_facts(topic) if topic else []

    # 构建提示
    facts_summary = "\n".join([
        f"  - {f['entity']}: {f['attribute']} = {f['value']} (置信度: {f['confidence']:.2f})"
        for f in facts[:20]  # 限制数量
    ])

    prompt = f"""
基于以下已核实的事实，请生成 3-4 个互斥的未来走向预测假说：

相关事实：
{facts_summary if facts_summary else "暂无相关事实"}

预测主题：{topic}

要求：
1. 生成 3-4 个互斥的假说
2. 每个假说都应具体且可验证
3. 格式：用 JSON 数组返回，每个元素包含 "id" 和 "description" 字段
"""

    if llm_client:
        response = llm_client.generate_sync(prompt)
        try:
            import json
            hypotheses_data = json.loads(response)
            hypotheses = [
                Hypothesis(id=h["id"], description=h["description"])
                for h in hypotheses_data
            ]
        except (json.JSONDecodeError, KeyError):
            # 解析失败，使用默认假说
            hypotheses = _generate_default_hypotheses(topic)
    else:
        hypotheses = _generate_default_hypotheses(topic)

    # 验证互斥性（简化版）
    validated = []
    for h in hypotheses:
        is_mutually_exclusive = True
        for vh in validated:
            if _is_overlapping(h.description, vh.description):
                is_mutually_exclusive = False
                break
        if is_mutually_exclusive:
            validated.append(h)

    return validated[:4]


def _generate_default_hypotheses(topic: str) -> list[Hypothesis]:
    """生成默认假说（当 LLM 不可用时）"""
    return [
        Hypothesis(id="A", description=f"敌方将在{topic}领域实现重大突破，提前于预期完成目标"),
        Hypothesis(id="B", description=f"敌方在{topic}领域进展缓慢，面临技术瓶颈，目标推迟"),
        Hypothesis(id="C", description=f"敌方在{topic}领域采取替代策略，绕过当前技术路线"),
        Hypothesis(id="D", description=f"敌方在{topic}领域遇到意外 setbacks，项目可能取消"),
    ]


def _is_overlapping(desc1: str, desc2: str) -> bool:
    """判断两个假说描述是否重叠（简化版）"""
    # 简单的关键词重叠检测
    keywords1 = set(desc1.split())
    keywords2 = set(desc2.split())
    overlap = keywords1 & keywords2
    return len(overlap) > len(keywords1) * 0.5


def build_evidence_matrix(hypotheses: list[Hypothesis],
                          evidence_set: list[dict],
                          llm_client: LLMClient = None) -> dict:
    """
    构建证据矩阵

    对每个证据评估其对每个假说的支持程度。
    """
    matrix = {}

    for h in hypotheses:
        matrix[h.id] = {}

        for evidence in evidence_set:
            eval_result = _evaluate_evidence_for_hypothesis(
                evidence, h, llm_client
            )
            matrix[h.id][evidence.get('id', str(id(evidence)))] = eval_result

    return matrix


def _evaluate_evidence_for_hypothesis(evidence: dict, hypothesis: Hypothesis,
                                       llm_client: LLMClient = None) -> str:
    """评估单个证据对假说的支持程度"""
    if llm_client:
        prompt = f"""
评估以下证据对假说的支持程度：

假说：{hypothesis.description}
证据：{evidence.get('content', evidence.get('description', str(evidence)))}

请选择：support / contradict / neutral / single_source

- support: 证据支持该假说
- contradict: 证据反驳该假说
- neutral: 证据与假说无关
- single_source: 证据仅来自单一来源（可疑）
"""
        response = llm_client.generate_sync(prompt).strip().lower()
        if response in ["support", "contradict", "neutral", "single_source"]:
            return response

    # 简化版评估逻辑
    evidence_content = str(evidence.get('content', evidence.get('description', '')))
    hypothesis_desc = hypothesis.description

    # 检查关键词匹配
    evidence_keywords = set(evidence_content.lower().split())
    hypothesis_keywords = set(hypothesis_desc.lower().split())

    overlap = evidence_keywords & hypothesis_keywords
    if len(overlap) > 2:
        return "support"
    elif len(overlap) > 0:
        return "neutral"
    else:
        return "neutral"


def calculate_probability_distribution(evidence_matrix: dict) -> dict:
    """
    计算假说概率分布

    使用加权计分法：
    score = support_count * 1.0 - contradict_count * 1.5 - single_source_count * 0.5
    """
    scores = {}

    for hypothesis_id, evidence_evals in evidence_matrix.items():
        support_count = sum(1 for v in evidence_evals.values() if v == "support")
        contradict_count = sum(1 for v in evidence_evals.values() if v == "contradict")
        single_source_count = sum(1 for v in evidence_evals.values() if v == "single_source")

        score = support_count * 1.0 - contradict_count * 1.5 - single_source_count * 0.5
        scores[hypothesis_id] = max(0, score)

    # 归一化
    total_score = sum(scores.values())
    if total_score > 0:
        probabilities = {h_id: scores[h_id] / total_score for h_id in scores}
    else:
        # 均匀分布
        n = len(scores)
        probabilities = {h_id: 1.0 / n for h_id in scores}

    return probabilities


def ach_analysis(topic: str, knowledge_graph: KnowledgeGraph,
                 llm_client: LLMClient = None) -> dict:
    """
    ACH 竞争性假说分析完整流程

    1. 生成假说
    2. 提取证据
    3. 构建证据矩阵
    4. 计算概率分布
    """
    # 1. 生成假说
    hypotheses = generate_hypotheses(topic, knowledge_graph, llm_client)

    # 2. 提取证据
    evidence_set = knowledge_graph.get_entity_facts(topic)
    if not evidence_set:
        # 如果没有直接相关的事实，获取全部事实
        evidence_set = knowledge_graph.get_entity_facts("")

    # 3. 构建证据矩阵
    evidence_matrix = build_evidence_matrix(hypotheses, evidence_set, llm_client)

    # 4. 计算概率分布
    probabilities = calculate_probability_distribution(evidence_matrix)

    # 5. 构建结果
    results = []
    for h in hypotheses:
        supporting = [e for e, v in evidence_matrix[h.id].items() if v == "support"]
        contradicting = [e for e, v in evidence_matrix[h.id].items() if v == "contradict"]
        single_source = [e for e, v in evidence_matrix[h.id].items() if v == "single_source"]

        results.append({
            "hypothesis": h.description,
            "probability": probabilities.get(h.id, 0.0),
            "supporting_evidence_count": len(supporting),
            "contradicting_evidence_count": len(contradicting),
            "single_source_evidence_count": len(single_source),
            "confidence": _calculate_hypothesis_confidence(
                len(supporting), len(contradicting), len(single_source)
            )
        })

    # 排序
    results.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "topic": topic,
        "hypotheses": results,
        "confidence": sum(r["confidence"] * r["probability"] for r in results),
        "evidence_count": len(evidence_set)
    }


def _calculate_hypothesis_confidence(support: int, contradict: int, single_source: int) -> float:
    """计算假说置信度"""
    if support == 0 and contradict == 0:
        return 0.0
    return max(0.0, min(1.0, (support - contradict * 1.5) / max(1, support + contradict + single_source * 0.5)))


# =============================================================================
# 6. TRL 轨迹预测算法
# =============================================================================

class TRLLevel(Enum):
    """技术成熟度等级 (1-9)"""
    TRL_1 = 1  # 基本原理观察
    TRL_2 = 2  # 技术原理发现
    TRL_3 = 3  # 定性定量验证
    TRL_4 = 4  # 实验验证
    TRL_5 = 5  # 实验环境验证
    TRL_6 = 6  # 相关环境原型验证
    TRL_7 = 7  # 外场原型验证
    TRL_8 = 8  # 实际系统验证
    TRL_9 = 9  # 作战应用


def assess_trl(evidence_set: list[dict]) -> dict:
    """
    评估技术成熟度 (TRL)

    基于不同类型证据的权重计算 TRL 等级。
    """
    trl_scores = {i: 0.0 for i in range(1, 10)}

    for evidence in evidence_set:
        evidence_type = evidence.get('type', 'unknown')
        confidence = evidence.get('confidence', 0.5)

        if evidence_type == "basic_research":
            trl_scores[1] += confidence * 0.3
            trl_scores[2] += confidence * 0.2
        elif evidence_type == "lab_experiment":
            trl_scores[3] += confidence * 0.4
            trl_scores[4] += confidence * 0.3
        elif evidence_type == "prototype":
            trl_scores[5] += confidence * 0.4
            trl_scores[6] += confidence * 0.3
        elif evidence_type == "field_test":
            trl_scores[7] += confidence * 0.4
            trl_scores[8] += confidence * 0.3
        elif evidence_type == "deployment":
            trl_scores[9] += confidence * 0.5
            trl_scores[8] += confidence * 0.2

    # 确定最终 TRL
    max_trl = max(trl_scores, key=trl_scores.get)
    total_score = sum(trl_scores.values())
    confidence = trl_scores[max_trl] / total_score if total_score > 0 else 0.0

    return {
        "trl": max_trl,
        "confidence": confidence,
        "scores": trl_scores
    }


def analyze_budget_trend(budget_history: list[dict]) -> dict:
    """
    分析预算趋势

    使用线性回归预测趋势，并检测突变。
    """
    if len(budget_history) < 2:
        return {"trend": "unknown", "slope": 0, "r_squared": 0, "significant_change": False}

    # 准备数据
    years = [h.get('year', i) for i, h in enumerate(budget_history)]
    amounts = [h.get('amount', 0) for h in budget_history]

    # 线性回归
    n = len(years)
    sum_x = sum(years)
    sum_y = sum(amounts)
    sum_xy = sum(x * y for x, y in zip(years, amounts))
    sum_x2 = sum(x * x for x in years)

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        slope = 0
        intercept = amounts[0]
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

    # R 平方
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in amounts)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(years, amounts))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 突变检测
    diffs = [amounts[i] - amounts[i-1] for i in range(1, len(amounts))]
    significant_change = any(
        abs(d) / max(abs(amounts[i-1]), 1) > 0.2
        for i, d in enumerate(diffs, 1)
    )

    return {
        "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
        "slope": slope,
        "r_squared": r_squared,
        "significant_change": significant_change,
        "yearly_data": [{"year": y, "amount": a} for y, a in zip(years, amounts)]
    }


def detect_weak_signals(event_stream: list[dict],
                        baseline_periods: int = 5) -> list[dict]:
    """
    弱信号检测算法

    检测事件频率异常、新关联等信号。
    """
    signals = []

    # 1. 频率分析
    event_type_counts = defaultdict(list)
    for event in event_stream:
        event_type = event.get('type', 'unknown')
        timestamp = event.get('timestamp')
        if timestamp:
            event_type_counts[event_type].append(timestamp)

    # 2. 异常频率检测
    for event_type, timestamps in event_type_counts.items():
        if len(timestamps) < baseline_periods + 1:
            continue

        # 按时间窗口统计
        sorted_ts = sorted(timestamps)
        baseline_counts = [1] * (len(sorted_ts) - 1)  # 简化：每次出现为 1
        recent_count = 1

        baseline_avg = sum(baseline_counts) / len(baseline_counts)
        if baseline_avg > 0 and recent_count > baseline_avg * 2:
            signals.append({
                "signal": f"异常频率: {event_type}",
                "strength": recent_count / baseline_avg,
                "confidence": 0.7,
                "type": "frequency_anomaly"
            })

    # 3. 关联分析（简化版）
    event_pairs = defaultdict(int)
    for i, event1 in enumerate(event_stream):
        for event2 in event_stream[i+1:]:
            if abs(event1.get('timestamp', 0) - event2.get('timestamp', 0)) < 3600:  # 1小时内
                pair_key = f"{event1.get('type', 'unknown')} ↔ {event2.get('type', 'unknown')}"
                event_pairs[pair_key] += 1

    for pair, count in event_pairs.items():
        if count >= 3:  # 至少 3 次共现
            signals.append({
                "signal": f"新关联: {pair}",
                "strength": count,
                "confidence": 0.8,
                "type": "correlation"
            })

    return signals


def forecast_trajectory(topic: str, knowledge_graph: KnowledgeGraph,
                        memory: TimeSeriesMemory = None,
                        llm_client: LLMClient = None) -> dict:
    """
    TRL 轨迹预测完整流程

    结合 TRL 评估、预算趋势和弱信号检测进行预测。
    """
    # 1. TRL 评估
    evidence_set = knowledge_graph.get_entity_facts(topic)
    trl_result = assess_trl(evidence_set)

    # 2. 预算趋势分析
    budget_history = memory.query(key="budget", value=topic) if memory else []
    budget_trend = analyze_budget_trend([
        {"year": i, "amount": e.metadata.get('amount', 0)}
        for i, e in enumerate(budget_history)
    ]) if budget_history else {"trend": "unknown", "slope": 0, "r_squared": 0, "significant_change": False}

    # 3. 弱信号检测
    event_stream = memory.query(key="event", value=topic) if memory else []
    weak_signals = detect_weak_signals([
        {"type": e.content, "timestamp": e.timestamp.timestamp()}
        for e in event_stream
    ]) if event_stream else []

    # 4. 结合 ACH 分析
    ach_result = ach_analysis(topic, knowledge_graph, llm_client)

    # 5. 综合预测
    return {
        "topic": topic,
        "trl": trl_result,
        "budget_trend": budget_trend,
        "weak_signals": weak_signals,
        "ach_analysis": ach_result,
        "overall_confidence": (
            trl_result["confidence"] * 0.3
            + (1 - abs(budget_trend.get("slope", 0)) / 1000) * 0.2
            + ach_result.get("confidence", 0.5) * 0.5
        )
    }
