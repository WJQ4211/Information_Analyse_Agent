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
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd="/".join(__file__.split("/")[:-1])
    )
    sys.exit(result.returncode)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="情报分析智能体 - 战略情报分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
    python main.py --mode demo                    # 运行演示
    python main.py --mode analyze --topic "导弹发展"  # 运行分析
    python main.py --mode test                    # 运行测试
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["analyze", "demo", "test"],
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

    args = parser.parse_args()

    if args.mode == "demo":
        asyncio.run(run_demo())
    elif args.mode == "analyze":
        result = asyncio.run(run_analysis(args.topic, use_demo=False))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[*] 结果已保存到: {args.output}")
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
