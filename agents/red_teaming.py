"""
情报分析智能体 - 多智能体红队对抗框架
Intelligence Analysis Agent - Multi-Agent Red Teaming Framework

包含：
1. Blue 智能体 (情报综合)
2. Red 智能体 (魔鬼代言人)
3. Judge 智能体 (指挥官仲裁)
4. 红队对抗循环算法
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.infrastructure import KnowledgeGraph, LLMClient
from core.algorithms import (
    Vulnerability, DOTMLPFPDimension,
    assess_evidence, Evidence, SourceInfo, SourceReliability,
    cross_validate_fact, update_knowledge_graph,
    deduce_vulnerabilities, compose_vulnerabilities,
    ach_analysis, forecast_trajectory,
    detect_procurement_anomaly, detect_academic_anomaly, detect_social_anomaly
)


# =============================================================================
# 智能体角色枚举
# =============================================================================

class AgentRole(Enum):
    BLUE = "blue"      # 情报综合智能体
    RED = "red"        # 魔鬼代言人
    JUDGE = "judge"    # 指挥官智能体


# =============================================================================
# 结论数据结构
# =============================================================================

@dataclass
class IntelligenceConclusion:
    """情报结论"""
    main_points: list[str]
    evidence: list[dict]
    confidence: float
    open_questions: list[str]
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    trajectory_predictions: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Challenge:
    """挑战"""
    type: str  # "logic_flaw", "fact_gap", "single_source", "maskirovka"
    description: str
    severity: float  # [0, 1]
    required_evidence: list[str]


@dataclass
class ArbitrationResult:
    """仲裁结果"""
    status: str  # "approved", "supplement_required", "rejected"
    final_conclusion: Optional[IntelligenceConclusion] = None
    revised_confidence: float = 0.0
    required_evidence: list[str] = field(default_factory=list)
    resolved_challenges: list[Challenge] = field(default_factory=list)


# =============================================================================
# Blue 智能体 (情报综合智能体)
# =============================================================================

class BlueAgent:
    """
    Blue 智能体 - 情报综合智能体

    职责：汇总事实，提出"弱点"与"走向"的初步推断结论。

    System Prompt: "基于已核实图谱事实，使用 ACH 法输出敌方能力评估。"
    """

    def __init__(self, llm_client: LLMClient, knowledge_graph: KnowledgeGraph):
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph
        self.role = AgentRole.BLUE
        self.system_prompt = (
            "你是一个专业的情报分析专家。你的任务是基于已核实的知识图谱事实，"
            "使用 ACH（Analysis of Competing Hypotheses）方法输出敌方能力评估。"
            "你需要：1) 综合所有已核实的事实，2) 识别敌方薄弱点，3) 预测未来走向，"
            "4) 提出带有置信度的结论。"
            "请确保你的推理基于至少两个独立信息源的事实。"
        )

    async def analyze(self, topic: str) -> IntelligenceConclusion:
        """
        Blue 智能体分析流程

        1. 提取知识图谱事实
        2. 进行薄弱点推演 (DOTMLPF-P)
        3. 进行轨迹预测 (TRL + ACH)
        4. 生成初步结论
        """
        # 1. 提取事实
        facts = self.knowledge_graph.get_entity_facts(topic)

        # 2. 薄弱点推演
        vulnerabilities = deduce_vulnerabilities(self.knowledge_graph)
        composite_vulns = compose_vulnerabilities(vulnerabilities)

        # 3. 轨迹预测
        trajectory = await self._forecast_trajectory(topic)

        # 4. 生成结论
        conclusion = await self._generate_conclusion(
            topic, facts, vulnerabilities, composite_vulns, trajectory
        )

        return conclusion

    async def _forecast_trajectory(self, topic: str) -> dict:
        """预测未来走向"""
        return forecast_trajectory(topic, self.knowledge_graph, llm_client=self.llm_client)

    async def _generate_conclusion(self, topic: str, facts: list[dict],
                                    vulnerabilities: list[Vulnerability],
                                    composite_vulns: list[dict],
                                    trajectory: dict) -> IntelligenceConclusion:
        """生成结论"""
        # 构建提示
        facts_summary = "\n".join([
            f"  - {f['entity']}: {f['attribute']} = {f['value']} (置信度: {f['confidence']:.2f})"
            for f in facts[:20]
        ])

        vulns_summary = "\n".join([
            f"  - {v.dimension.value}: 薄弱度 {v.weakness_score:.2f} - {v.description}"
            for v in vulnerabilities[:5]
        ])

        prompt = f"""
基于以下已核实的事实、薄弱点分析和轨迹预测，生成敌方能力评估的初步结论。

预测主题：{topic}

已核实事实：
{facts_summary if facts_summary else "暂无直接相关事实"}

薄弱点分析：
{vulns_summary if vulns_summary else "暂未识别到显著薄弱点"}

轨迹预测：
  - TRL 等级: {trajectory.get('trl', {}).get('trl', 'N/A')}
  - 预算趋势: {trajectory.get('budget_trend', {}).get('trend', 'N/A')}
  - 弱信号: {len(trajectory.get('weak_signals', []))} 个

请输出结构化的结论，包括：
1. 主要结论（3-5 个要点）
2. 支持证据（列出事实来源）
3. 置信度（0-1 分数）
4. 待验证的问题（需要更多证据的领域）

格式要求：使用 JSON 格式返回。
"""

        if self.llm_client:
            response = await self.llm_client.generate(prompt, self.system_prompt)
            try:
                import json
                data = json.loads(response)
                return IntelligenceConclusion(
                    main_points=data.get('main_points', []),
                    evidence=data.get('evidence', []),
                    confidence=data.get('confidence', 0.5),
                    open_questions=data.get('open_questions', []),
                    vulnerabilities=vulnerabilities,
                    trajectory_predictions=trajectory
                )
            except json.JSONDecodeError:
                pass

        # 默认结论
        return IntelligenceConclusion(
            main_points=[
                f"基于当前事实，敌方在{topic}领域存在一定的发展潜力",
                f"识别到 {len(vulnerabilities)} 个潜在薄弱点",
                f"轨迹预测显示 TRL 等级为 {trajectory.get('trl', {}).get('trl', 'N/A')}"
            ],
            evidence=[{"content": f, "confidence": 0.5} for f in facts[:5]],
            confidence=0.5,
            open_questions=["需要更多交叉印证的证据"],
            vulnerabilities=vulnerabilities,
            trajectory_predictions=trajectory
        )

    async def supplement(self, conclusion: IntelligenceConclusion,
                         required_evidence: list[str]) -> IntelligenceConclusion:
        """
        补充证据

        当 Judge 智能体要求补充证据时，Blue 智能体会尝试获取更多信息。
        """
        # 尝试从知识图谱中获取更多相关事实
        for evidence_request in required_evidence:
            # 查询更多相关信息
            additional_facts = self.knowledge_graph.query(entity=evidence_request)
            for fact in additional_facts:
                # 添加到结论的证据中
                conclusion.evidence.append({
                    "content": f"{fact.name}: {fact.attributes}",
                    "confidence": fact.confidence,
                    "source": fact.sources[0] if fact.sources else "unknown"
                })

        # 降低置信度（因为需要补充证据意味着初始结论不够可靠）
        conclusion.confidence *= 0.7

        return conclusion


# =============================================================================
# Red 智能体 (魔鬼代言人)
# =============================================================================

class RedAgent:
    """
    Red 智能体 - 魔鬼代言人

    职责：专门负责挑刺，寻找推断中的逻辑漏洞或事实缺失。

    System Prompt: "你是一个持怀疑态度的资深反情报官。你的任务是推翻 Blue 的结论，
    找出其论据中可能是敌方'战略欺骗'或孤证的部分。"
    """

    def __init__(self, llm_client: LLMClient, knowledge_graph: KnowledgeGraph):
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph
        self.role = AgentRole.RED
        self.system_prompt = (
            "你是一个持怀疑态度的资深反情报官。你的任务是推翻 Blue 智能体的结论，"
            "找出其论据中可能存在的问题，包括：\n"
            "1. 逻辑漏洞（逻辑推理错误）\n"
            "2. 事实缺失（缺少交叉印证）\n"
            "3. 单一来源（仅有一个信息来源）\n"
            "4. 战略欺骗（可能是敌方的 Maskirovka）\n"
            "请详细列出每个挑战，并说明严重程度和需要补充的证据。"
        )

    async def challenge(self, conclusion: IntelligenceConclusion) -> list[Challenge]:
        """
        Red 智能体挑战流程

        1. 检测逻辑漏洞
        2. 检测事实缺失
        3. 检测单一来源
        4. 检测战略欺骗
        """
        challenges = []

        # 1. 逻辑漏洞检测
        logic_challenges = await self._detect_logic_flaws(conclusion)
        challenges.extend(logic_challenges)

        # 2. 事实缺失检测
        fact_gap_challenges = await self._detect_fact_gaps(conclusion)
        challenges.extend(fact_gap_challenges)

        # 3. 单一来源检测
        single_source_challenges = await self._detect_single_source(conclusion)
        challenges.extend(single_source_challenges)

        # 4. 战略欺骗检测
        maskirovka_challenges = await self._detect_maskirovka(conclusion)
        challenges.extend(maskirovka_challenges)

        return challenges

    async def _detect_logic_flaws(self, conclusion: IntelligenceConclusion) -> list[Challenge]:
        """检测逻辑漏洞"""
        challenges = []

        # 检查结论之间的一致性
        if len(conclusion.main_points) > 1:
            for i, point1 in enumerate(conclusion.main_points):
                for j, point2 in enumerate(conclusion.main_points[i+1:], i+1):
                    if await self._is_contradictory(point1, point2):
                        challenges.append(Challenge(
                            type="logic_flaw",
                            description=f"结论 {i+1} 与结论 {j+1} 存在矛盾：'{point1}' vs '{point2}'",
                            severity=0.8,
                            required_evidence=["澄清两个结论之间的关系"]
                        ))

        # 检查因果链是否完整
        if len(conclusion.evidence) < 2:
            challenges.append(Challenge(
                type="logic_flaw",
                description="结论缺乏足够的证据支持（少于 2 条独立证据）",
                severity=0.9,
                required_evidence=["至少 2 条来自独立来源的证据"]
            ))

        return challenges

    async def _detect_fact_gaps(self, conclusion: IntelligenceConclusion) -> list[Challenge]:
        """检测事实缺失"""
        challenges = []

        # 检查每个结论是否有足够的支撑证据
        for point in conclusion.main_points:
            # 使用 LLM 分析该结论需要什么证据
            prompt = f"""
分析以下结论需要什么类型的证据来支持：

结论：{point}

请列出 2-3 种关键证据类型，这些证据应该来自独立的来源。
"""
            if self.llm_client:
                response = await self.llm_client.generate(prompt, self.system_prompt)
                # 解析响应（简化版）
                evidence_types = [line.strip() for line in response.split('\n') if line.strip()][:3]
            else:
                evidence_types = ["独立来源交叉印证", "历史数据对比", "专家意见"]

            challenges.append(Challenge(
                type="fact_gap",
                description=f"结论 '{point[:50]}...' 缺少关键证据",
                severity=0.6,
                required_evidence=evidence_types
            ))

        return challenges

    async def _detect_single_source(self, conclusion: IntelligenceConclusion) -> list[Challenge]:
        """检测单一来源"""
        challenges = []

        for evidence in conclusion.evidence:
            # 检查证据是否有多个来源
            sources = evidence.get('sources', [])
            if len(sources) < 2:
                challenges.append(Challenge(
                    type="single_source",
                    description=f"证据 '{str(evidence.get('content', ''))[:50]}...' 仅来自单一来源",
                    severity=0.5,
                    required_evidence=["来自不同独立来源的交叉印证"]
                ))

        return challenges

    async def _detect_maskirovka(self, conclusion: IntelligenceConclusion) -> list[Challenge]:
        """检测战略欺骗"""
        challenges = []

        # 检查结论中的事实是否与知识图谱中的历史事实冲突
        for evidence in conclusion.evidence:
            entity = evidence.get('entity', '')
            attribute = evidence.get('attribute', '')
            value = evidence.get('value')

            if entity and attribute and value:
                validation = cross_validate_fact(
                    entity, attribute, value, self.knowledge_graph
                )

                if validation.get('maskirovka_flag'):
                    challenges.append(Challenge(
                        type="maskirovka",
                        description=f"事实 '{entity}.{attribute} = {value}' 可能是战略欺骗",
                        severity=0.9,
                        required_evidence=["权威来源的交叉印证", "历史数据的对比分析"]
                    ))

        return challenges

    async def _is_contradictory(self, point1: str, point2: str) -> bool:
        """判断两个结论是否矛盾"""
        if not self.llm_client:
            return False

        prompt = f"""
判断以下两个结论是否矛盾：

结论 1：{point1}
结论 2：{point2}

如果矛盾，请回答 "YES"，否则回答 "NO"。
"""
        response = await self.llm_client.generate(prompt, self.system_prompt)
        return "YES" in response.upper()


# =============================================================================
# Judge 智能体 (指挥官智能体)
# =============================================================================

class JudgeAgent:
    """
    Judge 智能体 - 指挥官智能体

    职责：仲裁争论，要求 Blue 补充证据，或最终生成包含置信度的终版情报。

    System Prompt: "权衡双方辩点。剔除证据不足的推断，仅输出具备高置信度事实支撥的结论。"
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.role = AgentRole.JUDGE
        self.system_prompt = (
            "你是一个经验丰富的指挥官。你的任务是仲裁 Blue 和 Red 智能体之间的争论。\n"
            "请权衡双方的辩点，评估每个挑战的有效性，并做出以下判断：\n"
            "1. 如果挑战有效，需要 Blue 智能体补充证据\n"
            "2. 如果挑战无效，维持 Blue 的结论\n"
            "3. 最终输出具备高置信度事实支撑的结论\n"
            "请确保你的判断基于证据链原则（至少两个独立信息源）。"
        )

    async def arbitrate(self, conclusion: IntelligenceConclusion,
                        challenges: list[Challenge]) -> ArbitrationResult:
        """
        Judge 智能体仲裁流程

        1. 评估每个挑战的有效性
        2. 决定是否需要补充证据
        3. 生成最终结论
        """
        resolved_challenges = []
        supplement_required = False
        required_evidence = []

        for challenge in challenges:
            validity = await self._assess_challenge_validity(challenge, conclusion)

            if validity > 0.7:
                # 挑战有效，需要补充证据
                supplement_required = True
                required_evidence.extend(challenge.required_evidence)
                resolved_challenges.append(challenge)
            elif validity > 0.4:
                # 挑战部分有效，降低置信度
                conclusion.confidence *= 0.7
                resolved_challenges.append(challenge)
            else:
                # 挑战无效，维持结论
                pass

        if supplement_required:
            return ArbitrationResult(
                status="supplement_required",
                revised_confidence=conclusion.confidence * 0.5,
                required_evidence=list(set(required_evidence)),
                resolved_challenges=resolved_challenges
            )
        else:
            return ArbitrationResult(
                status="approved",
                final_conclusion=conclusion,
                revised_confidence=conclusion.confidence,
                resolved_challenges=resolved_challenges
            )

    async def _assess_challenge_validity(self, challenge: Challenge,
                                          conclusion: IntelligenceConclusion) -> float:
        """
        评估挑战的有效性

        基于挑战类型和严重程度计算有效性分数。
        """
        # 基础有效性 = 严重程度
        base_validity = challenge.severity

        # 根据挑战类型调整
        if challenge.type == "logic_flaw":
            # 逻辑漏洞通常较有效
            base_validity *= 1.0
        elif challenge.type == "fact_gap":
            # 事实缺失需要进一步评估
            base_validity *= 0.8
        elif challenge.type == "single_source":
            # 单一来源是中等有效
            base_validity *= 0.7
        elif challenge.type == "maskirovka":
            # 战略欺骗是高度有效的挑战
            base_validity *= 1.2

        # 考虑结论的现有置信度
        if conclusion.confidence > 0.8:
            # 高置信度的结论更难被推翻
            base_validity *= 0.8
        elif conclusion.confidence < 0.5:
            # 低置信度的结论更容易被推翻
            base_validity *= 1.2

        return min(1.0, base_validity)


# =============================================================================
# 红队对抗循环算法
# =============================================================================

class RedTeamOrchestrator:
    """
    红队对抗编排器

    协调 Blue、Red、Judge 三个智能体进行多轮博弈。
    """

    def __init__(self, blue_agent: BlueAgent, red_agent: RedAgent,
                 judge_agent: JudgeAgent, max_rounds: int = 3):
        self.blue_agent = blue_agent
        self.red_agent = red_agent
        self.judge_agent = judge_agent
        self.max_rounds = max_rounds

    async def debate(self, topic: str) -> dict:
        """
        红队对抗循环算法

        1. Blue 智能体生成初步结论
        2. Red 智能体提出挑战
        3. Judge 智能体仲裁
        4. 如果需要补充，Blue 智能体补充证据
        5. 重复直到达到最大轮次或结论被批准
        """
        # 1. Blue 生成初步结论
        conclusion = await self.blue_agent.analyze(topic)

        # 2. 循环博弈
        round_num = 0
        all_challenges = []

        while round_num < self.max_rounds:
            # Red 挑战
            challenges = await self.red_agent.challenge(conclusion)
            all_challenges.extend(challenges)

            # Judge 仲裁
            arbitration = await self.judge_agent.arbitrate(conclusion, challenges)

            if arbitration.status == "approved":
                break
            elif arbitration.status == "supplement_required":
                # Blue 补充证据
                conclusion = await self.blue_agent.supplement(
                    conclusion, arbitration.required_evidence
                )
                round_num += 1
            else:
                break

        # 3. 返回最终结果
        final_conclusion = arbitration.final_conclusion or conclusion

        return {
            "topic": topic,
            "final_conclusion": {
                "main_points": final_conclusion.main_points,
                "confidence": final_conclusion.confidence,
                "evidence_count": len(final_conclusion.evidence),
                "open_questions": final_conclusion.open_questions
            },
            "debate_rounds": round_num,
            "total_challenges": len(all_challenges),
            "resolved_challenges": [
                {
                    "type": c.type,
                    "description": c.description,
                    "severity": c.severity
                }
                for c in arbitration.resolved_challenges
            ],
            "vulnerabilities": [
                {
                    "dimension": v.dimension.value,
                    "weakness_score": v.weakness_score,
                    "confidence": v.confidence,
                    "description": v.description
                }
                for v in final_conclusion.vulnerabilities
            ],
            "timestamp": datetime.now().isoformat()
        }
