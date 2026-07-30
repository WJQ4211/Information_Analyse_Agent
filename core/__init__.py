"""
情报分析智能体 - 核心模块
Intelligence Analysis Agent - Core Module
"""

from core.infrastructure import (
    KnowledgeGraph,
    VectorDatabase,
    LLMClient,
    LLMConfig,
    TimeSeriesMemory,
    GraphNode,
    GraphEdge,
    MemoryEntry,
)

from core.algorithms import (
    # NATO 6×6
    SourceReliability,
    InfoCredibility,
    SourceInfo,
    Evidence,
    AssessmentResult,
    calculate_source_reliability,
    calculate_info_credibility,
    assess_evidence,

    # GraphRAG
    cross_validate_fact,
    update_knowledge_graph,

    # DOTMLPF-P
    DOTMLPFPDimension,
    DimensionEvidence,
    Vulnerability,
    score_dimension,
    deduce_vulnerabilities,
    compose_vulnerabilities,

    # Anomaly Detection
    detect_procurement_anomaly,
    detect_academic_anomaly,
    detect_social_anomaly,

    # ACH
    Hypothesis,
    EvidenceEvaluation,
    generate_hypotheses,
    build_evidence_matrix,
    calculate_probability_distribution,
    ach_analysis,

    # TRL
    TRLLevel,
    assess_trl,
    analyze_budget_trend,
    detect_weak_signals,
    forecast_trajectory,
)

__all__ = [
    # Infrastructure
    "KnowledgeGraph",
    "VectorDatabase",
    "LLMClient",
    "LLMConfig",
    "TimeSeriesMemory",
    "GraphNode",
    "GraphEdge",
    "MemoryEntry",

    # NATO 6×6
    "SourceReliability",
    "InfoCredibility",
    "SourceInfo",
    "Evidence",
    "AssessmentResult",
    "calculate_source_reliability",
    "calculate_info_credibility",
    "assess_evidence",

    # GraphRAG
    "cross_validate_fact",
    "update_knowledge_graph",

    # DOTMLPF-P
    "DOTMLPFPDimension",
    "DimensionEvidence",
    "Vulnerability",
    "score_dimension",
    "deduce_vulnerabilities",
    "compose_vulnerabilities",

    # Anomaly Detection
    "detect_procurement_anomaly",
    "detect_academic_anomaly",
    "detect_social_anomaly",

    # ACH
    "Hypothesis",
    "EvidenceEvaluation",
    "generate_hypotheses",
    "build_evidence_matrix",
    "calculate_probability_distribution",
    "ach_analysis",

    # TRL
    "TRLLevel",
    "assess_trl",
    "analyze_budget_trend",
    "detect_weak_signals",
    "forecast_trajectory",
]
