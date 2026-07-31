"""
情报分析智能体 - LLM 提示词模板
Intelligence Analysis Agent - LLM Prompt Templates

定义实体抽取和关系抽取所需的 LLM 提示词。
这些提示词经过精心设计，确保 LLM 输出结构化的 JSON 格式。
"""


# =============================================================================
# 实体类型描述 (Entity Type Descriptions)
# =============================================================================

ENTITY_TYPE_DESCRIPTIONS = """
实体类型说明：
- Weapon: 武器系统（导弹、战斗机、舰船等），属性包括射程、速度、数量、服役状态等
- Organization: 军事/政府/科研机构，属性包括角色、成立时间、人员规模等
- Person: 关键人物（指挥官、科学家、决策者），属性包括职务、专业领域等
- Event: 军事/政治事件（试验、部署、冲突、演习），属性包括时间、地点、规模等
- Location: 地理位置（基地、试验场、部署区域），属性包括坐标、类型、用途等
- Technology: 技术/子系统（雷达、制导、通信），属性包括成熟度、性能参数等
- Budget: 预算/财务记录，属性包括金额、年度、用途分类等
"""


# =============================================================================
# 关系类型描述 (Relation Type Descriptions)
# =============================================================================

RELATION_TYPE_DESCRIPTIONS = """
关系类型说明：
- developed_by: 武器/技术由某机构研发
- deployed_at: 武器/部队部署在某地点
- depends_on: 系统依赖某技术/资源/供应商
- commanded_by: 部队/单位由某人指挥
- funded_by: 项目由某预算/机构资助
- tested_at: 武器/系统在试验场进行测试
- has_issue: 系统/装备存在已知问题
- requires: 系统需要某训练/维护/资源
- located_in: 地点位于某更大区域
- competes_with: 项目/技术与替代方案竞争
- participates_in: 人员/单位参与某事件
- succeeded_by: 项目/系统被后续版本替代
"""


# =============================================================================
# 实体抽取提示词 (Entity Extraction Prompt)
# =============================================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的军事情报实体抽取专家。你的任务是从文本中提取结构化的实体信息。

要求：
1. 只提取文本中明确提及的实体，不要推测或编造
2. 每个实体必须有唯一的 entity_id（格式：类型_名称的拼音或英文简写）
3. 对每个实体评估抽取置信度（0-1 之间）
4. 属性值应为具体数值或事实，不要笼统描述
5. 严格输出 JSON 格式，不要附加任何解释文字"""

ENTITY_EXTRACTION_PROMPT = """从以下军事情报文本中提取所有实体。

{entity_type_descriptions}

文本内容：
\"\"\"
{text}
\"\"\"

请输出 JSON 格式：
{{
  "entities": [
    {{
      "entity_id": "weapon_df21d",
      "entity_type": "Weapon",
      "name": "DF-21D 反舰弹道导弹",
      "attributes": {{
        "range_km": 1500,
        "speed_mach": 3.0,
        "status": "服役中"
      }},
      "confidence": 0.9
    }}
  ]
}}

如果没有找到任何实体，请输出：{{"entities": []}}"""


# =============================================================================
# 关系抽取提示词 (Relation Extraction Prompt)
# =============================================================================

RELATION_EXTRACTION_SYSTEM_PROMPT = """你是一个专业的军事情报关系抽取专家。你的任务是从文本中识别实体之间的关系。

要求：
1. 只提取文本中明确表述的关系，不要推测
2. source_entity 和 target_entity 必须是已识别实体的 entity_id 或 name
3. 对每条关系评估置信度（0-1 之间）
4. 严格输出 JSON 格式，不要附加任何解释文字
5. 注意：一条文本中可能包含多条关系，请全部提取"""

RELATION_EXTRACTION_PROMPT = """基于以下已识别的实体，从文本中提取它们之间的关系。

{relation_type_descriptions}

已识别的实体：
{entities_summary}

文本内容：
\"\"\"
{text}
\"\"\"

示例输出格式：
{{
  "relations": [
    {{
      "source_entity": "weapon_df21d",
      "target_entity": "org_casic",
      "relation_type": "developed_by",
      "confidence": 0.85
    }},
    {{
      "source_entity": "weapon_df21d",
      "target_entity": "location_hainan",
      "relation_type": "deployed_at",
      "confidence": 0.75
    }},
    {{
      "source_entity": "weapon_df21d",
      "target_entity": "tech_aesa_radar",
      "relation_type": "depends_on",
      "confidence": 0.70
    }}
  ]
}}

请输出 JSON 格式：
{{
  "relations": [
    {{
      "source_entity": "...",
      "target_entity": "...",
      "relation_type": "...",
      "confidence": 0.0
    }}
  ]
}}

如果没有找到任何关系，请输出：{{"relations": []}}"""
