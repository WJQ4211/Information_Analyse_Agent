#!/usr/bin/env python3
"""
情报分析智能体 - 主入口
Intelligence Analysis Agent - Main Entry Point

使用方法：
    python main.py [--topic TOPIC] [--mode MODE]

模式：
    --mode analyze: 运行完整分析流程
    --mode demo: 运行演示（使用模拟数据）
    --mode test: 运行单元测试
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta

from core import (
    KnowledgeGraph,
    LLMClient,
    LLMConfig,
    TimeSeriesMemory,
    GraphNode,
    GraphEdge,
    SourceInfo,
    SourceReliability,
    Evidence,
    assess_evidence,
    cross_validate_fact,
    update_knowledge_graph,
    deduce_vulnerabilities,
    compose_vulnerabilities,
    ach_analysis,
    forecast_trajectory,
    detect_procurement_anomaly,
    detect_academic_anomaly,
    detect_social_anomaly,
)
from agents import (
    BlueAgent,
    RedAgent,
    JudgeAgent,
    RedTeamOrchestrator,
)
from ingestion import (
    IngestionPipeline,
    IngestionResult,
)


def create_demo_knowledge_graph() -> KnowledgeGraph:
    """创建演示用的知识图谱（模拟敌方武器发展情报）"""
    kg = KnowledgeGraph()

    # 添加实体：新型导弹系统
    missile_node = GraphNode(
        id="DF-21D_Missile",
        type="Weapon",
        name="DF-21D 反舰弹道导弹",
        attributes={
            "range_km": 1500,
            "speed_mach": 3.0,
            "warhead_type": "活爆头",
            "launch_platform": "陆基",
            "deployed_units": 72,
            "development_start": "2015-01-01",
            "first_flight": "2017-06-01",
            "entered_service": "2019-01-01"
        },
        confidence=0.9,
        sources=["简氏防务周刊", "国防部公报"]
    )
    kg.add_node(missile_node)

    # 添加实体：研发单位
    org_node = GraphNode(
        id="China_Aerospace_Science_and_Industry",
        type="Organization",
        name="中国航天工业集团",
        attributes={
            "role": "主要研发单位",
            "established": "1950-10-01",
            "employees": 150000
        },
        confidence=0.85,
        sources=["官方网站", "行业报告"]
    )
    kg.add_node(org_node)

    # 添加关系
    kg.add_edge(GraphEdge(
        source_id="DF-21D_Missile",
        target_id="China_Aerospace_Science_and_Industry",
        relation="developed_by",
        confidence=0.9
    ))

    # 添加采购记录
    procurement_node = GraphNode(
        id="procurement_2024_001",
        type="Event",
        name="2024年批量采购",
        attributes={
            "category": "元器件",
            "amount": 5000000,
            "supplier": "进口供应商",
            "timestamp": "2024-03-15T00:00:00"
        },
        confidence=0.7,
        sources=["招标公告"]
    )
    kg.add_node(procurement_node)

    kg.add_edge(GraphEdge(
        source_id="DF-21D_Missile",
        target_id="procurement_2024_001",
        relation="depends_on",
        confidence=0.7
    ))

    # 添加维修相关社交媒体帖子
    social_node = GraphNode(
        id="social_complaint_001",
        type="Event",
        name="部队维修抱怨",
        attributes={
            "content": "缺少维修手册，维护周期过长",
            "source_type": "社交媒体",
            "timestamp": "2024-05-20T00:00:00"
        },
        confidence=0.6,
        sources=["军事论坛"]
    )
    kg.add_node(social_node)

    kg.add_edge(GraphEdge(
        source_id="DF-21D_Missile",
        target_id="social_complaint_001",
        relation="has_issue",
        confidence=0.6
    ))

    # 添加训练维度
    training_node = GraphNode(
        id="training_program_001",
        type="Training",
        name="操作培训计划",
        attributes={
            "status": "不足",
            "trained_personnel": 30,
            "required_personnel": 72,
            "issue": "训练资料缺乏"
        },
        confidence=0.75,
        sources=["内部报告"]
    )
    kg.add_node(training_node)

    kg.add_edge(GraphEdge(
        source_id="DF-21D_Missile",
        target_id="training_program_001",
        relation="requires",
        confidence=0.75
    ))

    return kg


def create_demo_memory() -> TimeSeriesMemory:
    """创建演示用的时序记忆库"""
    memory = TimeSeriesMemory(max_entries=1000)

    # 添加预算历史
    base_date = datetime(2020, 1, 1)
    for i in range(5):
        memory.add(
            content=f"年度预算记录",
            metadata={
                "key": "budget",
                "value": "DF-21D",
                "amount": 5000000 + i * 1000000,
                "year": 2020 + i
            },
            timestamp=base_date + timedelta(days=365 * i)
        )

    # 添加事件历史
    events = [
        ("2020-06-01", "实验验证完成"),
        ("2021-03-15", "原型测试成功"),
        ("2022-08-20", "外场试验"),
        ("2023-01-10", "正式入列"),
        ("2024-03-15", "批量采购"),
    ]

    for date_str, content in events:
        memory.add(
            content=content,
            metadata={
                "key": "event",
                "value": "DF-21D",
                "type": "milestone"
            },
            timestamp=datetime.fromisoformat(date_str)
        )

    return memory


async def run_analysis(topic: str, use_demo: bool = True) -> dict:
    """
    运行完整的情报分析流程

    1. 初始化知识图谱和 LLM 客户端
    2. 创建多智能体
    3. 运行红队对抗
    4. 输出结果
    """
    print(f"[*] 开始分析主题: {topic}")
    print(f"[*] 模式: {'演示' if use_demo else '生产'}")

    # 1. 初始化
    if use_demo:
        kg = create_demo_knowledge_graph()
        memory = create_demo_memory()
        llm = LLMClient(LLMConfig())  # 使用模拟 LLM
    else:
        kg = KnowledgeGraph()
        memory = TimeSeriesMemory()
        llm = LLMClient(LLMConfig(
            model_name="qwen2-72b",
            api_base="http://localhost:8000/v1"
        ))

    print("[*] 知识图谱初始化完成")

    # 2. 创建智能体
    blue = BlueAgent(llm, kg)
    red = RedAgent(llm, kg)
    judge = JudgeAgent(llm)
    orchestrator = RedTeamOrchestrator(blue, red, judge, max_rounds=3)

    print("[*] 智能体初始化完成")

    # 3. 运行红队对抗
    print("[*] 开始红队对抗...")
    result = await orchestrator.debate(topic)

    print("[*] 红队对抗完成")

    # 4. 输出结果
    print("\n" + "=" * 60)
    print("情报分析结果")
    print("=" * 60)

    print(f"\n主题: {result['topic']}")
    print(f"置信度: {result['final_conclusion']['confidence']:.2f}")
    print(f"对抗轮次: {result['debate_rounds']}")
    print(f"总挑战数: {result['total_challenges']}")

    print("\n主要结论:")
    for i, point in enumerate(result['final_conclusion']['main_points'], 1):
        print(f"  {i}. {point}")

    print("\n薄弱点分析:")
    for vuln in result['vulnerabilities']:
        print(f"  - {vuln['dimension']}: 薄弱度 {vuln['weakness_score']:.2f}")
        print(f"    {vuln['description']}")

    print("\n解决的挑战:")
    for challenge in result['resolved_challenges']:
        print(f"  - [{challenge['type']}] {challenge['description']}")

    print("\n" + "=" * 60)

    return result


async def run_ingest(
    sources: list[str],
    source_name: str,
    source_reliability: str,
    use_demo: bool = True,
) -> dict:
    """
    运行 OSINT 数据采集与知识图谱构建

    Args:
        sources: 数据源列表，格式为 "type:path"（如 "web:https://...", "pdf:./file.pdf"）
        source_name: 来源名称（用于 NATO 评级）
        source_reliability: 来源可靠度等级 (A-F)
        use_demo: 是否使用 Mock LLM
    """
    print(f"[*] 开始数据采集与知识图谱构建")
    print(f"[*] 数据源数量: {len(sources)}")
    print(f"[*] 来源名称: {source_name}")
    print(f"[*] 可靠度等级: {source_reliability}")

    # 1. 初始化基础设施
    if use_demo:
        kg = KnowledgeGraph()
        memory = TimeSeriesMemory(max_entries=1000)
        llm = LLMClient(LLMConfig())
    else:
        kg = KnowledgeGraph()
        memory = TimeSeriesMemory()
        llm = LLMClient(LLMConfig(
            model_name="qwen2-72b",
            api_base="http://localhost:8000/v1",
        ))

    print("[*] 基础设施初始化完成")

    # 2. 构建来源信息
    reliability_map = {
        "A": SourceReliability.A, "B": SourceReliability.B,
        "C": SourceReliability.C, "D": SourceReliability.D,
        "E": SourceReliability.E, "F": SourceReliability.F,
    }
    reliability = reliability_map.get(source_reliability.upper(), SourceReliability.C)

    source_info = SourceInfo(
        name=source_name,
        reliability=reliability,
        history_accuracy=0.7,
        expertise_level=0.7,
        independence=0.6,
        traceability=True,
    )

    # 3. 创建管道
    pipeline = IngestionPipeline(kg, llm, memory=memory)

    # 4. 分类数据源
    urls: list[str] = []
    files: list[str] = []

    for source_str in sources:
        parts = source_str.split(":", 1)
        if len(parts) != 2:
            print(f"[!] 格式错误，跳过: {source_str}（应为 type:path）")
            continue

        source_type, source_path = parts
        if source_type == "pdf":
            files.append(source_path)
        elif source_type in ("web", "feed"):
            urls.append(source_str)  # 保留 type:path 格式供管道解析
        else:
            print(f"[!] 未知类型，跳过: {source_type}")

    # 5. 执行采集
    print("[*] 开始数据采集...")
    result = IngestionResult()

    if urls:
        print(f"[*] 采集 URL: {len(urls)} 个")
        url_result = await pipeline.ingest_from_urls(
            [s.split(":", 1)[1] for s in urls],
            source_info,
        )
        result.merge(url_result)

    if files:
        print(f"[*] 处理文件: {len(files)} 个")
        file_result = await pipeline.ingest_from_files(files, source_info)
        result.merge(file_result)

    # 6. 输出结果
    print("\n" + "=" * 60)
    print("采集与入库结果")
    print("=" * 60)
    print(f"  文档数: {result.document_count}")
    print(f"  实体数: {result.entity_count}")
    print(f"  关系数: {result.relation_count}")
    print(f"  冲突数: {result.conflict_count}")
    print(f"  战略欺骗嫌疑: {'是' if result.has_maskirovka else '否'}")
    print(f"  耗时: {result.duration_seconds:.2f} 秒")
    print(f"  状态: {result.status.value}")

    if result.warnings:
        print("\n警告信息:")
        for w in result.warnings:
            print(f"  [!] {w}")

    if result.conflicts:
        print("\n冲突记录:")
        for c in result.conflicts:
            flag = " [战略欺骗]" if c.maskirovka_flag else ""
            print(f"  - {c.entity_name}.{c.attribute}: 冲突度 {c.conflict_score:.2f}{flag}")

    # 7. 知识图谱统计
    print(f"\n知识图谱统计:")
    print(f"  节点数: {len(kg.nodes)}")
    print(f"  边数: {len(kg.edges)}")

    if kg.nodes:
        print("\n实体列表:")
        for node_id, node in kg.nodes.items():
            print(f"  [{node.type}] {node.name} (置信度: {node.confidence:.2f})")

    print("\n" + "=" * 60)

    # 8. 保存结果
    output = {
        "ingestion_result": {
            "document_count": result.document_count,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
            "conflict_count": result.conflict_count,
            "has_maskirovka": result.has_maskirovka,
            "duration_seconds": result.duration_seconds,
            "status": result.status.value,
            "warnings": result.warnings,
            "conflicts": [
                {
                    "entity": c.entity_name,
                    "attribute": c.attribute,
                    "conflict_score": c.conflict_score,
                    "maskirovka": c.maskirovka_flag,
                }
                for c in result.conflicts
            ],
        },
        "knowledge_graph": {
            "node_count": len(kg.nodes),
            "edge_count": len(kg.edges),
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "name": n.name,
                    "confidence": n.confidence,
                    "sources": n.sources,
                }
                for n in kg.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation,
                    "confidence": e.confidence,
                }
                for e in kg.edges
            ],
        },
        "timestamp": datetime.now().isoformat(),
    }

    return output


async def run_demo() -> None:
    """运行完整演示"""
    print("=" * 60)
    print("情报分析智能体 - 演示模式")
    print("=" * 60)
    print()

    # 演示 1: NATO 6×6 评估
    print("[演示 1] NATO 6×6 惑评估")
    print("-" * 40)

    source = SourceInfo(
        name="简氏防务周刊",
        reliability=SourceReliability.A,
        history_accuracy=0.95,
        expertise_level=0.9,
        independence=0.8,
        traceability=True
    )

    evidence = Evidence(
        content="DF-21D 导弹射程 1500 公里",
        source=source,
        timestamp=datetime.now(),
        entity="DF-21D",
        attribute="range_km",
        value=1500,
        cross_references=["国防部公报", "军事专家分析"]
    )

    assessment = assess_evidence(evidence)
    print(f"  源可靠度: {assessment.source_reliability.value}")
    print(f"  信息可信度: {assessment.info_credibility.value}")
    print(f"  综合置信度: {assessment.combined_confidence:.2f}")
    print(f"  置信等级: {assessment.confidence_level}")
    print(f"  战略欺骗: {'是' if assessment.maskirovka_flag else '否'}")
    print()

    # 演示 2: 完整分析流程
    print("[演示 2] 完整情报分析流程")
    print("-" * 40)
    result = await run_analysis("DF-21D 反舰弹道导弹发展", use_demo=True)

    # 保存结果
    output_file = "analysis_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[*] 结果已保存到: {output_file}")


def run_tests() -> None:
    """运行单元测试"""
    import subprocess
    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=project_dir,
    )
    sys.exit(result.returncode)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="情报分析智能体 - 战略情报分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python main.py --mode demo                              # 运行演示
    python main.py --mode analyze --topic "导弹发展"            # 运行分析（需要本地 LLM）
    python main.py --mode ingest --sources "web:https://example.com" --source-name "简氏防务" --source-reliability A
    python main.py --mode ingest --sources "web:https://example.com" --demo  # 使用 Mock LLM 测试
    python main.py --mode test                              # 运行测试
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["analyze", "demo", "test", "ingest"],
        default="demo",
        help="运行模式"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="DF-21D 反舰弹道导弹发展",
        help="分析主题"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_result.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--sources", "-s",
        type=str,
        nargs="+",
        default=[],
        help="数据源列表（格式: type:path），支持 web/feed/pdf"
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="OSINT 公开来源",
        help="来源名称（用于 NATO 评级）"
    )
    parser.add_argument(
        "--source-reliability",
        type=str,
        default="C",
        choices=["A", "B", "C", "D", "E", "F"],
        help="来源可靠度等级 (A=极其可靠, F=无法评定)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="使用 Mock LLM 运行（用于测试，不需要本地 LLM 服务）"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_demo())
    elif args.mode == "analyze":
        result = asyncio.run(run_analysis(args.topic, use_demo=False))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存到: {args.output}")
    elif args.mode == "ingest":
        if not args.sources:
            print("[!] 错误: ingest 模式需要指定 --sources 参数")
            print("    示例: python main.py --mode ingest --sources \"web:https://example.com\"")
            sys.exit(1)
        result = asyncio.run(run_ingest(
            sources=args.sources,
            source_name=args.source_name,
            source_reliability=args.source_reliability,
            use_demo=args.demo,
        ))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存到: {args.output}")
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
