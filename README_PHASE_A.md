# 分歧驱动的持续研判原型：阶段 A

这是《分歧驱动的国防科技情报持续研判：最小可运行研究原型实现设计》的阶段 A 实现。

阶段 A 只做以下事情：

- 用合成证据导入并冻结 T0/T1 证据快照；
- 用手工 JSON 冻结 2 条竞争性假设；
- 用手工 JSON 写入 4 个观点、2 个分歧和监测计划；
- 通过证据依赖、触发器和关键词兜底扫描定位 T1 受影响观点；
- 只为受影响观点追加新版本，保留未受影响观点的原版本；
- 生成可反查证据与版本的 Markdown 报告和运行清单；
- 用 pytest 验证快照不可变、引用完整、版本追加和局部更新。

本阶段不导入真实案例、不联网、不调用大模型，也不实现前端。`synthetic: true` 是示例对象的强制标记，示例输出不应作为论文案例结论使用。

## 快速运行

在本目录执行：

```text
python -m src.pipeline run-synthetic
pytest
```

等价脚本入口：`python scripts/01_init_db.py` 建库/迁移，`python scripts/02_run_phase_a_demo.py` 运行合成验收流程。

演示命令使用 `fixtures/approvals/` 中预置的合成审核记录来逐个通过 G1～G4。它们只是状态机测试夹具，不代表真实专家签发。

也可以按设计文档中的命令逐步运行：

```text
python -m src.pipeline init-case --config config/case_synthetic.yaml
python -m src.pipeline ingest --case synthetic_military_ai --phase t0 --input data/raw/t0
python -m src.pipeline freeze --case synthetic_military_ai --phase t0 --approve-file fixtures/approvals/g1_t0.json
python -m src.pipeline hypotheses --case synthetic_military_ai --snapshot <T0快照ID> --input fixtures/hypotheses_t0.json --approve-file fixtures/approvals/g2.json
python -m src.pipeline analyze-t0 --case synthetic_military_ai --snapshot <T0快照ID> --input-dir fixtures/analysis_t0 --approve-monitor-file fixtures/approvals/g3.json --approve-report-file fixtures/approvals/g4_t0.json
python -m src.pipeline ingest --case synthetic_military_ai --phase t1 --input data/raw/t1
python -m src.pipeline freeze --case synthetic_military_ai --phase t1 --approve-file fixtures/approvals/g1_t1.json
python -m src.pipeline update-t1 --case synthetic_military_ai --from <T0快照ID> --to <T1快照ID> --input fixtures/update_t1.json --approve-affected-file fixtures/approvals/affected_t1.json --approve-report-file fixtures/approvals/g4_t1.json
```

默认数据库为 `data/research.sqlite3`，运行产物位于 `outputs/<run_id>/`。可以用 `--workspace` 指定另一份工作目录。

## 明确未实现内容

阶段 B 的 `ModelClient`、提示模板、模型调用、结构化输出重试、真实材料抽取和语义检索均未实现。完成阶段 A 验收前不会接入这些内容。
