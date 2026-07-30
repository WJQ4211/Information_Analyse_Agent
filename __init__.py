"""
情报分析智能体
Intelligence Analysis Agent

一个基于 NATO 6×6、GraphRAG、DOTMLPF-P、ACH 和 TRL 技术的战略情报分析系统。

主要功能：
1. OSINT 数据的事实提取与核验
2. 敌方薄弱点推演
3. 未来走向预测
4. 多智能体红队对抗

使用方法：
    from core import KnowledgeGraph, LLMClient, LLMConfig
    from agents import BlueAgent, RedAgent, JudgeAgent, RedTeamOrchestrator

    # 初始化
    kg = KnowledgeGraph()
    llm = LLMClient(LLMConfig())

    # 创建智能体
    blue = BlueAgent(llm, kg)
    red = RedAgent(llm, kg)
    judge = JudgeAgent(llm)
    orchestrator = RedTeamOrchestrator(blue, red, judge)

    # 运行分析
    result = await orchestrator.debate("敌方新型武器发展")
"""

__version__ = "1.0.0"
__author__ = "Intelligence Analysis Agent Team"
