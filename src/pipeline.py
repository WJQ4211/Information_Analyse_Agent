"""Command-line orchestration for the no-LLM Phase A prototype."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import yaml

from .schemas import (
    DiscriminativeNeed,
    Disagreement,
    Evidence,
    EvidenceCitation,
    EvidenceDependency,
    EvidenceSnapshot,
    HumanReview,
    Hypothesis,
    Importance,
    MonitorIndicator,
    SnapshotTag,
    TriggerRule,
    UpdateEvent,
    Viewpoint,
    ViewpointStatus,
    DependencyRelation,
    evidence_ref,
    model_dump,
    model_validate,
    model_copy,
    stable_hash,
)
from .storage import StorageError, Store


class GatePending(RuntimeError):
    """Raised when an explicit gate approval artifact was not supplied."""

    def __init__(self, path: Path, gate: str) -> None:
        super().__init__(f"{gate} approval is required; pending file written to {path}")
        self.path = path
        self.gate = gate


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id(prefix: str = "phase-a") -> str:
    return f"{prefix}-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"


def workspace_path(workspace: str | Path) -> Path:
    return Path(workspace).resolve()


def store_for(workspace: str | Path) -> Store:
    root = workspace_path(workspace)
    return Store(root / "data" / "research.sqlite3")


def output_dir(workspace: str | Path, run_id: str) -> Path:
    path = workspace_path(workspace) / "outputs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            data = model_dump(value) if hasattr(value, "dict") or hasattr(value, "model_dump") else value
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path: str | Path, key: Optional[str] = None) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".json", ".jsonl"})
        records: List[Dict[str, Any]] = []
        for file in files:
            records.extend(read_records(file, key=key))
        return records
    if source.suffix.lower() == ".jsonl":
        records = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Each JSONL line must be an object: {source}")
                records.append(value)
        return records
    value = read_json(source)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if key and isinstance(value.get(key), list):
            return value[key]
        for candidate in ("evidence", "hypotheses", "viewpoints", "disagreements", "needs", "indicators", "triggers"):
            if isinstance(value.get(candidate), list):
                return value[candidate]
        return [value]
    raise ValueError(f"Expected a JSON object/list: {source}")


def read_object(path: Path, key: str) -> List[Dict[str, Any]]:
    return read_records(path, key=key) if path.exists() else []


def load_yaml(path: str | Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML config must be an object: {path}")
    return value


def load_case_config(store: Store, case_id: str, config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    config = store.get_case_config(case_id)
    if config is None and config_path:
        config = load_yaml(config_path)
        store.save_case_config(case_id, config)
    if config is None:
        raise StorageError(f"Case is not initialized: {case_id}")
    return config


def parse_cutoff(config: Mapping[str, Any], phase: SnapshotTag, override: Optional[str]) -> datetime:
    text = override or config.get("cutoffs", {}).get(phase.value.lower())
    if not text:
        raise ValueError(f"No cutoff configured for {phase.value}")
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def save_manifest(
    workspace: str | Path,
    run_id: str,
    case_id: str,
    *,
    stage: str,
    artifacts: Mapping[str, str],
    snapshots: Sequence[str] = (),
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    root = workspace_path(workspace)
    out = output_dir(root, run_id)
    manifest: Dict[str, Any] = {
        "manifest_version": "phase-a.1",
        "run_id": run_id,
        "case_id": case_id,
        "stage": stage,
        "mode": "synthetic_manual_json" if "synthetic" in case_id else "manual_json",
        "model_client": None,
        "snapshots": list(snapshots),
        "artifacts": dict(artifacts),
        "reproducibility": {
            "python": sys.version.split()[0],
            "llm_called": False,
            "input_hash": stable_hash(dict(artifacts)),
        },
    }
    if extra:
        manifest.update(extra)
    path = out / "manifest.json"
    write_json(path, manifest)
    return path


def approval(
    store: Store,
    workspace: str | Path,
    *,
    case_id: str,
    gate: str,
    artifact: str,
    approval_file: Optional[str | Path],
    run_id: str,
) -> Dict[str, Any]:
    if not approval_file:
        pending = output_dir(workspace, run_id) / "reviews" / f"{gate}_pending.json"
        write_json(
            pending,
            {
                "gate": gate,
                "case_id": case_id,
                "approved": False,
                "reviewer": "",
                "artifact": artifact,
                "notes": "Fill this file and pass it back with the gate-specific approval option.",
            },
        )
        raise GatePending(pending, gate)
    data = read_json(approval_file)
    if not isinstance(data, dict) or not data.get("approved"):
        raise StorageError(f"{gate} approval file is not approved: {approval_file}")
    reviewer = str(data.get("reviewer", "")).strip()
    if not reviewer:
        raise StorageError(f"{gate} approval must contain reviewer: {approval_file}")
    review = HumanReview(
        review_id=f"REV-{gate}-{stable_hash(data)[:12]}",
        gate=gate,
        case_id=case_id,
        approved=True,
        reviewer=reviewer,
        artifact=artifact,
        notes=str(data.get("notes", "")),
        run_id=run_id,
    )
    store.record_human_review(review)
    return data


def init_case(workspace: str | Path, config_path: str | Path, run_id: Optional[str] = None) -> Dict[str, Any]:
    config = load_yaml(config_path)
    case_id = str(config["case_id"])
    root = workspace_path(workspace)
    (root / "data" / "raw" / "t0").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw" / "t1").mkdir(parents=True, exist_ok=True)
    (root / "data" / "snapshots").mkdir(parents=True, exist_ok=True)
    with Store(root / "data" / "research.sqlite3") as store:
        store.save_case_config(case_id, config)
        actual_run = run_id or new_run_id("init")
        store.record_model_run(actual_run, case_id, "A", "manual_synthetic", {"config": str(config_path)})
        config_artifact = root / "data" / "cases" / f"{case_id}.json"
        write_json(config_artifact, config)
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={"case_config": str(config_artifact.relative_to(root))},
            extra={"synthetic": bool(config.get("synthetic", False))},
        )
    return {"case_id": case_id, "run_id": actual_run, "manifest": str(manifest)}


def ingest(
    workspace: str | Path,
    *,
    case_id: str,
    phase: SnapshotTag,
    input_path: str | Path,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id("ingest")
    with Store(root / "data" / "research.sqlite3") as store:
        load_case_config(store, case_id)
        records = read_records(input_path, key="evidence")
        inserted: List[Evidence] = []
        for raw in records:
            data = dict(raw)
            data["snapshot_tag"] = phase.value
            if "run_id" not in data:
                data["run_id"] = actual_run
            evidence = model_validate(Evidence, data)
            inserted.append(store.insert_evidence(evidence))  # type: ignore[arg-type]
        store.record_model_run(
            actual_run,
            case_id,
            phase.value,
            "manual_synthetic" if all(e.synthetic for e in inserted) else "manual_json",
            {"input": str(input_path), "count": len(inserted), "llm_called": False},
        )
        out = output_dir(root, actual_run)
        write_jsonl(out / f"evidence_{phase.value.lower()}.jsonl", inserted)
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={"ingested_evidence": str((out / f"evidence_{phase.value.lower()}.jsonl").relative_to(root))},
            extra={"phase": phase.value, "evidence_count": len(inserted)},
        )
    return {"case_id": case_id, "run_id": actual_run, "count": len(inserted), "manifest": str(manifest)}


def freeze_snapshot(
    workspace: str | Path,
    *,
    case_id: str,
    phase: SnapshotTag,
    approval_file: Optional[str | Path],
    cutoff: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id(f"freeze-{phase.value.lower()}")
    with Store(root / "data" / "research.sqlite3") as store:
        config = load_case_config(store, case_id)
        approved = approval(
            store,
            root,
            case_id=case_id,
            gate=f"G1_{phase.value}",
            artifact=f"evidence_snapshot_{phase.value}",
            approval_file=approval_file,
            run_id=actual_run,
        )
        cutoff_time = parse_cutoff(config, phase, cutoff)
        tags = [SnapshotTag.T0] if phase == SnapshotTag.T0 else [SnapshotTag.T0, SnapshotTag.T1]
        evidence = [e for e in store.list_evidence(tags=tags) if e.published_at <= cutoff_time.date()]
        refs = [evidence_ref(e.evidence_id, e.version) for e in evidence]
        if not refs:
            raise StorageError(f"No evidence available for {phase.value} before {cutoff_time.isoformat()}")
        snapshot_id = f"SNAP-{case_id}-{phase.value}-{stable_hash(refs)[:12]}"
        snapshot = store.create_snapshot(
            snapshot_id=snapshot_id,
            case_id=case_id,
            phase=phase,
            cutoff_time=cutoff_time,
            evidence_versions=refs,
            frozen_by=str(approved.get("reviewer")),
            run_id=actual_run,
        )
        store.record_model_run(actual_run, case_id, phase.value, "manual_synthetic", {"snapshot_id": snapshot_id})
        snapshot_file = root / "data" / "snapshots" / f"{snapshot_id}.json"
        write_json(snapshot_file, model_dump(snapshot))
        out = output_dir(root, actual_run)
        write_json(out / "evidence_snapshot.json", model_dump(snapshot))
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={"evidence_snapshot": str(snapshot_file.relative_to(root))},
            snapshots=[snapshot.snapshot_id],
            extra={"phase": phase.value, "gate": f"G1_{phase.value}"},
        )
    return {"case_id": case_id, "run_id": actual_run, "snapshot_id": snapshot.snapshot_id, "manifest": str(manifest)}


def freeze_hypotheses(
    workspace: str | Path,
    *,
    case_id: str,
    snapshot_id: str,
    input_path: str | Path,
    approval_file: Optional[str | Path],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id("hypotheses")
    with Store(root / "data" / "research.sqlite3") as store:
        snapshot = store.get_snapshot(snapshot_id)
        if not snapshot or snapshot.phase != SnapshotTag.T0:
            raise StorageError("Hypotheses must be frozen against an existing T0 snapshot")
        approval(
            store,
            root,
            case_id=case_id,
            gate="G2",
            artifact="hypotheses_frozen",
            approval_file=approval_file,
            run_id=actual_run,
        )
        hypotheses: List[Hypothesis] = []
        for raw in read_records(input_path, key="hypotheses"):
            data = dict(raw)
            data.update({"snapshot_id": snapshot_id, "frozen": True, "run_id": actual_run})
            hypotheses.append(model_validate(Hypothesis, data))  # type: ignore[arg-type]
        if not hypotheses:
            raise StorageError("At least one frozen hypothesis is required")
        store.insert_hypotheses(hypotheses)
        store.record_model_run(actual_run, case_id, "T0", "manual_synthetic", {"hypotheses": len(hypotheses)})
        out = output_dir(root, actual_run)
        write_json(out / "hypotheses_frozen.json", [model_dump(h) for h in hypotheses])
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={"hypotheses": str((out / "hypotheses_frozen.json").relative_to(root))},
            snapshots=[snapshot_id],
            extra={"hypothesis_count": len(hypotheses), "gate": "G2"},
        )
    return {"case_id": case_id, "run_id": actual_run, "snapshot_id": snapshot_id, "count": len(hypotheses), "manifest": str(manifest)}


def _citation_relation(citation: EvidenceCitation) -> DependencyRelation:
    return citation.relation


def _auto_dependencies(viewpoints: Sequence[Viewpoint], run_id: str) -> List[EvidenceDependency]:
    edges: List[EvidenceDependency] = []
    for viewpoint in viewpoints:
        for index, citation in enumerate(viewpoint.supporting_evidence):
            edges.append(
                EvidenceDependency(
                    edge_id=f"EDGE-{viewpoint.viewpoint_id}-{viewpoint.version}-S{index + 1}",
                    evidence_id=citation.evidence_id,
                    evidence_version=citation.version,
                    viewpoint_id=viewpoint.viewpoint_id,
                    viewpoint_version=viewpoint.version,
                    target_field="supporting_evidence",
                    relation=DependencyRelation.SUPPORT,
                    importance=Importance.HIGH if index == 0 else Importance.MEDIUM,
                    created_by=run_id,
                    run_id=run_id,
                )
            )
        for index, citation in enumerate(viewpoint.counter_evidence):
            edges.append(
                EvidenceDependency(
                    edge_id=f"EDGE-{viewpoint.viewpoint_id}-{viewpoint.version}-C{index + 1}",
                    evidence_id=citation.evidence_id,
                    evidence_version=citation.version,
                    viewpoint_id=viewpoint.viewpoint_id,
                    viewpoint_version=viewpoint.version,
                    target_field="counter_evidence",
                    relation=DependencyRelation.CHALLENGE,
                    importance=Importance.MEDIUM,
                    created_by=run_id,
                    run_id=run_id,
                )
            )
    return edges


def analyze_t0(
    workspace: str | Path,
    *,
    case_id: str,
    snapshot_id: str,
    input_dir: str | Path,
    monitor_approval_file: Optional[str | Path],
    report_approval_file: Optional[str | Path],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id("analyze-t0")
    source = Path(input_dir)
    with Store(root / "data" / "research.sqlite3") as store:
        snapshot = store.get_snapshot(snapshot_id)
        if not snapshot or snapshot.phase != SnapshotTag.T0:
            raise StorageError("T0 analysis requires a frozen T0 snapshot")
        hypotheses = store.list_hypotheses(snapshot_id)
        if not hypotheses:
            raise StorageError("T0 analysis requires frozen hypotheses")
        approval(
            store,
            root,
            case_id=case_id,
            gate="G3",
            artifact="monitor_plan",
            approval_file=monitor_approval_file,
            run_id=actual_run,
        )
        viewpoints: List[Viewpoint] = []
        for raw in read_records(source / "viewpoints.json", key="viewpoints"):
            data = dict(raw)
            data.update({"snapshot_id": snapshot_id, "run_id": actual_run})
            viewpoints.append(model_validate(Viewpoint, data))  # type: ignore[arg-type]
        expected = len(hypotheses) * 2
        if len(viewpoints) != expected:
            raise StorageError(f"Phase A fixture must cover every hypothesis in both perspectives ({expected} viewpoints expected)")
        hypothesis_ids = {h.hypothesis_id for h in hypotheses}
        if {v.hypothesis_id for v in viewpoints} != hypothesis_ids:
            raise StorageError("T0 viewpoints do not cover exactly the frozen hypotheses")
        for viewpoint in viewpoints:
            store.insert_viewpoint(viewpoint)
        dependencies = _load_dependencies(source / "dependencies.jsonl", actual_run)
        if not dependencies:
            dependencies = _auto_dependencies(viewpoints, actual_run)
        for dependency in dependencies:
            store.insert_dependency(dependency)
        disagreements: List[Disagreement] = []
        for raw in read_records(source / "disagreements.json", key="disagreements"):
            data = dict(raw)
            data.update({"snapshot_id": snapshot_id, "run_id": actual_run})
            disagreements.append(model_validate(Disagreement, data))  # type: ignore[arg-type]
            store.insert_disagreement(disagreements[-1])
        needs: List[DiscriminativeNeed] = []
        for raw in read_records(source / "needs.json", key="needs"):
            data = dict(raw)
            data["run_id"] = actual_run
            needs.append(model_validate(DiscriminativeNeed, data))  # type: ignore[arg-type]
            store.insert_need(needs[-1])
        indicators: List[MonitorIndicator] = []
        for raw in read_records(source / "indicators.json", key="indicators"):
            data = dict(raw)
            data.update({"run_id": actual_run, "status": "approved"})
            indicators.append(model_validate(MonitorIndicator, data))  # type: ignore[arg-type]
            store.insert_indicator(indicators[-1])
        triggers: List[TriggerRule] = []
        for raw in read_records(source / "triggers.json", key="triggers"):
            data = dict(raw)
            data.update({"run_id": actual_run, "status": "approved"})
            triggers.append(model_validate(TriggerRule, data))  # type: ignore[arg-type]
            store.insert_trigger(triggers[-1])
        store.record_model_run(actual_run, case_id, "T0", "manual_synthetic", {"llm_called": False})
        out = output_dir(root, actual_run)
        write_jsonl(out / "viewpoints_t0.jsonl", viewpoints)
        write_jsonl(out / "dependencies_t0.jsonl", dependencies)
        write_jsonl(out / "disagreements_t0.jsonl", disagreements)
        write_json(out / "monitor_plan.json", {"needs": [model_dump(n) for n in needs], "indicators": [model_dump(i) for i in indicators], "triggers": [model_dump(t) for t in triggers]})
        report = render_t0_report(store, case_id, snapshot, hypotheses, viewpoints, disagreements, needs, indicators, triggers)
        report_path = out / "report_t0.md"
        report_path.write_text(report, encoding="utf-8")
        approval(
            store,
            root,
            case_id=case_id,
            gate="G4_T0",
            artifact=str(report_path.relative_to(root)),
            approval_file=report_approval_file,
            run_id=actual_run,
        )
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={
                "viewpoints": str((out / "viewpoints_t0.jsonl").relative_to(root)),
                "dependencies": str((out / "dependencies_t0.jsonl").relative_to(root)),
                "disagreements": str((out / "disagreements_t0.jsonl").relative_to(root)),
                "monitor_plan": str((out / "monitor_plan.json").relative_to(root)),
                "report_t0": str(report_path.relative_to(root)),
            },
            snapshots=[snapshot_id],
            extra={"viewpoint_count": len(viewpoints), "disagreement_count": len(disagreements), "gate": "G3/G4_T0"},
        )
    return {"case_id": case_id, "run_id": actual_run, "snapshot_id": snapshot_id, "viewpoints": len(viewpoints), "disagreements": len(disagreements), "manifest": str(manifest)}


def _load_dependencies(path: Path, run_id: str) -> List[EvidenceDependency]:
    if not path.exists():
        return []
    return [model_validate(EvidenceDependency, {**raw, "created_by": raw.get("created_by", run_id), "run_id": run_id}) for raw in read_records(path)]  # type: ignore[list-item]


def render_t0_report(
    store: Store,
    case_id: str,
    snapshot: EvidenceSnapshot,
    hypotheses: Sequence[Hypothesis],
    viewpoints: Sequence[Viewpoint],
    disagreements: Sequence[Disagreement],
    needs: Sequence[DiscriminativeNeed],
    indicators: Sequence[MonitorIndicator],
    triggers: Sequence[TriggerRule],
) -> str:
    lines = [
        "# T0 条件化研判报告（阶段A合成数据）",
        "",
        "> 本报告由无大模型的手工 JSON 状态机生成，仅用于软件验收，不代表真实案例结论。",
        "",
        f"- case: `{case_id}`",
        f"- snapshot: `{snapshot.snapshot_id}`",
        f"- cutoff: `{snapshot.cutoff_time.isoformat()}`",
        f"- evidence manifest: `{snapshot.manifest_hash}`",
        "- 主判断签发：待 G4 人工填写；本阶段不自动替代专家判断。",
        "",
        "## 冻结假设",
        "",
    ]
    for hypothesis in hypotheses:
        lines.extend([f"### {hypothesis.hypothesis_id}", "", hypothesis.statement, "", f"- 可观察含义：{'；'.join(hypothesis.observable_implications)}", f"- 可证伪条件：{'；'.join(hypothesis.falsifiers)}", ""])
    lines.extend(["## 观点与证据", ""])
    for viewpoint in viewpoints:
        citations = ", ".join(f"`{c.ref}`" for c in viewpoint.supporting_evidence + viewpoint.counter_evidence)
        lines.extend([
            f"### {viewpoint.viewpoint_id} v{viewpoint.version}（{viewpoint.perspective_id}/{viewpoint.hypothesis_id}）",
            "",
            f"- stance: `{viewpoint.stance.value}`；status: `{viewpoint.status.value}`",
            f"- judgment: {viewpoint.judgment}",
            f"- evidence: {citations}",
            f"- key assumptions: {'；'.join(viewpoint.key_assumptions)}",
            f"- boundaries: {'；'.join(viewpoint.boundary_conditions)}",
            f"- falsifiers: {'；'.join(viewpoint.falsifiers)}",
            "",
        ])
    lines.extend(["## 未决分歧", ""])
    for disagreement in disagreements:
        lines.extend([
            f"- **{disagreement.disagreement_id}** `{disagreement.type.value}` / `{disagreement.decision_impact.value}`：{disagreement.contested_node}；处置：{disagreement.resolution_action}。",
            f"  - A: {disagreement.branch_a.claim}",
            f"  - B: {disagreement.branch_b.claim}",
        ])
    lines.extend(["", "## 判别性证据与监测计划", ""])
    for need in needs:
        lines.append(f"- **{need.need_id}**：{need.question}；判别性评分：{need.discriminativeness if need.discriminativeness is not None else '待人工填写'}。")
    for indicator in indicators:
        lines.append(f"  - 指标 `{indicator.indicator_id}`：{indicator.name}；A方向：{indicator.direction_a}；B方向：{indicator.direction_b}。")
    for trigger in triggers:
        lines.append(f"  - 触发器 `{trigger.trigger_id}`：{trigger.predicate_type.value} `{json.dumps(trigger.predicate, ensure_ascii=False)}`；动作：{trigger.action}。")
    lines.extend(["", "## 审计回链", "", "报告中的证据标识均以 `evidence_id:version` 形式引用；观点版本、证据依赖和人工审核记录保存在 SQLite 与运行产物中。", ""])
    return "\n".join(lines)


def build_monitor_plan(
    workspace: str | Path,
    *,
    case_id: str,
    snapshot_id: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id("monitor-plan")
    with Store(root / "data" / "research.sqlite3") as store:
        if not store.get_snapshot(snapshot_id):
            raise StorageError(f"Unknown snapshot: {snapshot_id}")
        needs = store.list_needs()
        indicators = store.list_indicators()
        triggers = store.list_triggers()
        out = output_dir(root, actual_run)
        plan_path = out / "monitor_plan.json"
        write_json(plan_path, {"needs": [model_dump(n) for n in needs], "indicators": [model_dump(i) for i in indicators], "triggers": [model_dump(t) for t in triggers]})
        manifest = save_manifest(root, actual_run, case_id, stage="A", artifacts={"monitor_plan": str(plan_path.relative_to(root))}, snapshots=[snapshot_id])
    return {"run_id": actual_run, "monitor_plan": str(plan_path), "manifest": str(manifest)}


def evidence_delta(store: Store, from_snapshot: EvidenceSnapshot, to_snapshot: EvidenceSnapshot) -> List[Evidence]:
    old = set(from_snapshot.evidence_versions)
    result: List[Evidence] = []
    for ref in sorted(set(to_snapshot.evidence_versions) - old):
        evidence_id, version = ref.rsplit(":", 1)
        evidence = store.get_evidence(evidence_id, int(version))
        if evidence:
            result.append(evidence)
    return result


def _evidence_text(evidence: Evidence) -> str:
    return " ".join([evidence.title, evidence.excerpt, evidence.normalized_claim, *evidence.topics]).lower()


def locate_affected(
    store: Store,
    *,
    from_snapshot: EvidenceSnapshot,
    to_snapshot: EvidenceSnapshot,
    viewpoints: Sequence[Viewpoint],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    delta = evidence_delta(store, from_snapshot, to_snapshot)
    candidates: Dict[str, List[str]] = {}
    triggered_by: Dict[str, List[str]] = {}
    dependencies = store.list_dependencies()
    triggers = store.list_triggers(approved_only=True)
    for evidence in delta:
        ref = evidence_ref(evidence.evidence_id, evidence.version)
        text = _evidence_text(evidence)
        for dependency in dependencies:
            if dependency.evidence_id == evidence.evidence_id and evidence.version >= dependency.evidence_version:
                if dependency.viewpoint_id in {v.viewpoint_id for v in viewpoints}:
                    reason = f"direct:{ref}:{dependency.relation.value}:{dependency.importance.value}"
                    candidates.setdefault(dependency.viewpoint_id, []).append(reason)
                    triggered_by.setdefault(dependency.viewpoint_id, []).append(ref)
        for trigger in triggers:
            if evidence.source_grade not in trigger.required_source_grade:
                continue
            terms = [str(t).lower() for t in trigger.predicate.get("event_terms", [])]
            if trigger.predicate_type.value == "event_match" and terms and any(term in text for term in terms):
                for viewpoint_id in trigger.affected_viewpoint_ids:
                    if viewpoint_id in {v.viewpoint_id for v in viewpoints}:
                        candidates.setdefault(viewpoint_id, []).append(f"trigger:{trigger.trigger_id}:{ref}")
                        triggered_by.setdefault(viewpoint_id, []).append(ref)
        # Conservative, high-recall safety net: use curated topic phrases only.
        # Full-text token overlap would make generic words such as "operator"
        # spuriously affect every operations viewpoint.
        for viewpoint in viewpoints:
            target = " ".join([
                viewpoint.judgment,
                *viewpoint.key_assumptions,
                *viewpoint.mechanism_claims,
                *viewpoint.boundary_conditions,
                *viewpoint.falsifiers,
            ]).lower()
            terms = [str(topic).lower() for topic in evidence.topics if len(str(topic)) >= 2]
            if any(term in target for term in terms):
                candidates.setdefault(viewpoint.viewpoint_id, []).append(f"safety_net:{ref}")
                triggered_by.setdefault(viewpoint.viewpoint_id, []).append(ref)
    for key in candidates:
        candidates[key] = sorted(set(candidates[key]))
        triggered_by[key] = sorted(set(triggered_by[key]))
    return candidates, triggered_by


def _set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", path)
    if not match:
        raise StorageError(f"Only top-level fields and list indexes may be patched: {path}")
    field, index = match.group(1), match.group(2)
    if field not in data:
        raise StorageError(f"Cannot patch unknown viewpoint field: {field}")
    if index is None:
        data[field] = value
        return
    position = int(index)
    if not isinstance(data[field], list) or position >= len(data[field]):
        raise StorageError(f"Cannot patch missing list item: {path}")
    data[field][position] = value


def apply_viewpoint_patch(
    store: Store,
    *,
    old: Viewpoint,
    patch: Mapping[str, Any],
    to_snapshot_id: str,
    run_id: str,
) -> Tuple[Viewpoint, List[str]]:
    changes = patch.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise StorageError(f"Patch for {old.viewpoint_id} must contain non-empty changes")
    changed_fields = list(patch.get("changed_fields", changes.keys()))
    if set(changed_fields) != set(changes.keys()):
        raise StorageError(f"changed_fields must exactly match changes for {old.viewpoint_id}")
    before = patch.get("before", {})
    old_data = model_dump(old)
    for field, expected in before.items():
        if field in old_data and old_data[field] != expected:
            raise StorageError(f"Patch before-value mismatch for {old.viewpoint_id}.{field}")
    new_data = dict(old_data)
    for path, value in changes.items():
        _set_path(new_data, path, value)
    new_data.update({
        "version": old.version + 1,
        "parent_version": old.version,
        "snapshot_id": to_snapshot_id,
        "change_reason": str(patch.get("change_reason", "")),
        "run_id": run_id,
        "created_at": utc_now().isoformat(),
        "generator": {"kind": "manual_synthetic_patch", "run_id": run_id, "llm_called": False},
    })
    if not new_data["change_reason"].strip():
        raise StorageError(f"Patch for {old.viewpoint_id} must include change_reason")
    for ref in patch.get("trigger_evidence", []):
        evidence_id, version = ref.rsplit(":", 1)
        if not store.snapshot_contains(to_snapshot_id, evidence_id, int(version)):
            raise StorageError(f"Trigger evidence is not in T1 snapshot: {ref}")
    new_viewpoint = model_validate(Viewpoint, new_data)
    store.insert_viewpoint(new_viewpoint)  # type: ignore[arg-type]
    return new_viewpoint, changed_fields


def render_t1_report(
    *,
    case_id: str,
    from_snapshot: EvidenceSnapshot,
    to_snapshot: EvidenceSnapshot,
    current_viewpoints: Sequence[Viewpoint],
    event: UpdateEvent,
    change_log: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# T1 局部更新报告（阶段A合成数据）",
        "",
        "> 本报告由无大模型的手工 JSON 局部更新状态机生成，仅用于软件验收。",
        "",
        f"- case: `{case_id}`",
        f"- from snapshot: `{from_snapshot.snapshot_id}`",
        f"- to snapshot: `{to_snapshot.snapshot_id}`",
        f"- updated viewpoints: {', '.join(f'`{v.viewpoint_id} v{v.version}`' for v in current_viewpoints if v.version > 1) or '无'}",
        f"- unchanged viewpoints: {', '.join(f'`{v}`' for v in event.unchanged_viewpoints) or '无'}",
        "",
        "## 当前观点版本",
        "",
    ]
    for viewpoint in current_viewpoints:
        citations = ", ".join(f"`{c.ref}`" for c in viewpoint.supporting_evidence + viewpoint.counter_evidence)
        lines.extend([
            f"### {viewpoint.viewpoint_id} v{viewpoint.version}",
            "",
            f"- stance: `{viewpoint.stance.value}`",
            f"- judgment: {viewpoint.judgment}",
            f"- evidence: {citations}",
            f"- version reason: {viewpoint.change_reason}",
            "",
        ])
    lines.extend(["## 字段级变化记录", ""])
    for change in change_log:
        lines.append(f"- `{change['viewpoint_id']}` v{change['old_version']}→v{change['new_version']}: {', '.join(change['changed_fields'])}; {change['change_reason']}")
    lines.extend(["", "## 影响定位路径", ""])
    for path, refs in event.triggered_by.items():
        lines.append(f"- {path}: {', '.join(f'`{ref}`' for ref in refs)}")
    lines.append("")
    return "\n".join(lines)


def update_t1(
    workspace: str | Path,
    *,
    case_id: str,
    from_snapshot_id: str,
    to_snapshot_id: str,
    input_path: str | Path,
    affected_approval_file: Optional[str | Path],
    report_approval_file: Optional[str | Path],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = workspace_path(workspace)
    actual_run = run_id or new_run_id("update-t1")
    with Store(root / "data" / "research.sqlite3") as store:
        config = load_case_config(store, case_id)
        from_snapshot = store.get_snapshot(from_snapshot_id)
        to_snapshot = store.get_snapshot(to_snapshot_id)
        if not from_snapshot or from_snapshot.phase != SnapshotTag.T0:
            raise StorageError("from snapshot must be a frozen T0 snapshot")
        if not to_snapshot or to_snapshot.phase != SnapshotTag.T1:
            raise StorageError("to snapshot must be a frozen T1 snapshot")
        if to_snapshot.cutoff_time <= from_snapshot.cutoff_time:
            raise StorageError("T1 cutoff must be later than T0 cutoff")
        baseline = store.list_viewpoints(from_snapshot_id)
        if not baseline:
            raise StorageError("No T0 viewpoints available for local update")
        current_by_id: Dict[str, Viewpoint] = {}
        for viewpoint in baseline:
            current_by_id[viewpoint.viewpoint_id] = viewpoint
        candidates, triggered_by = locate_affected(
            store,
            from_snapshot=from_snapshot,
            to_snapshot=to_snapshot,
            viewpoints=list(current_by_id.values()),
        )
        approval_data = approval(
            store,
            root,
            case_id=case_id,
            gate="G4_T1_AFFECTED",
            artifact="affected_viewpoints",
            approval_file=affected_approval_file,
            run_id=actual_run,
        )
        approved_affected = sorted(set(str(v) for v in approval_data.get("approved_affected", [])))
        unknown = set(approved_affected) - set(candidates)
        if unknown:
            raise StorageError(f"Human-approved affected viewpoints were not found by the three-path locator: {sorted(unknown)}")
        patches = {str(p["viewpoint_id"]): p for p in read_records(input_path, key="patches")}
        missing_patch = set(approved_affected) - set(patches)
        if missing_patch:
            raise StorageError(f"Missing manual update patches: {sorted(missing_patch)}")
        new_viewpoints: List[Viewpoint] = []
        change_log: List[Dict[str, Any]] = []
        for viewpoint_id in approved_affected:
            new_viewpoint, changed_fields = apply_viewpoint_patch(
                store,
                old=current_by_id[viewpoint_id],
                patch=patches[viewpoint_id],
                to_snapshot_id=to_snapshot_id,
                run_id=actual_run,
            )
            new_viewpoints.append(new_viewpoint)
            for citation in new_viewpoint.supporting_evidence + new_viewpoint.counter_evidence:
                relation = citation.relation
                dependency = EvidenceDependency(
                    edge_id=f"EDGE-{new_viewpoint.viewpoint_id}-{new_viewpoint.version}-{citation.evidence_id}-{citation.version}",
                    evidence_id=citation.evidence_id,
                    evidence_version=citation.version,
                    viewpoint_id=new_viewpoint.viewpoint_id,
                    viewpoint_version=new_viewpoint.version,
                    target_field="supporting_evidence" if citation in new_viewpoint.supporting_evidence else "counter_evidence",
                    relation=relation,
                    importance=Importance.HIGH if relation != DependencyRelation.BACKGROUND else Importance.LOW,
                    created_by=actual_run,
                    run_id=actual_run,
                )
                store.insert_dependency(dependency)
            change_log.append({
                "viewpoint_id": viewpoint_id,
                "old_version": current_by_id[viewpoint_id].version,
                "new_version": new_viewpoint.version,
                "trigger_evidence": patches[viewpoint_id].get("trigger_evidence", []),
                "changed_fields": changed_fields,
                "before": patches[viewpoint_id].get("before", {}),
                "after": patches[viewpoint_id].get("after", {}),
                "change_reason": new_viewpoint.change_reason,
                "status_transition": f"{current_by_id[viewpoint_id].status.value}->{new_viewpoint.status.value}",
            })
        unchanged = sorted(set(current_by_id) - set(approved_affected))
        event = UpdateEvent(
            update_id=f"UPD-{case_id}-{stable_hash([from_snapshot_id, to_snapshot_id, approved_affected, candidates])[:12]}",
            case_id=case_id,
            from_snapshot=from_snapshot_id,
            to_snapshot=to_snapshot_id,
            triggered_by={"delta_evidence": [evidence_ref(e.evidence_id, e.version) for e in evidence_delta(store, from_snapshot, to_snapshot)]},
            affected_candidates=candidates,
            approved_affected=approved_affected,
            new_viewpoint_versions=[f"{v.viewpoint_id}:{v.version}" for v in new_viewpoints],
            unchanged_viewpoints=unchanged,
            main_judgment_changed=False,
            change_summary=[f"{entry['viewpoint_id']} changed fields: {', '.join(entry['changed_fields'])}" for entry in change_log],
            run_id=actual_run,
        )
        # Keep the path-level map explicit in the persisted event for audit.
        event.triggered_by.update({f"paths:{key}": value for key, value in triggered_by.items()})
        store.insert_update_event(event)
        current = [store.latest_viewpoint(viewpoint_id) for viewpoint_id in sorted(current_by_id)]
        current_viewpoints = [v for v in current if v is not None]
        out = output_dir(root, actual_run)
        write_jsonl(out / "delta_evidence_t1.jsonl", evidence_delta(store, from_snapshot, to_snapshot))
        write_json(out / "affected_viewpoints.json", {"candidates": candidates, "approved_affected": approved_affected, "paths": triggered_by})
        write_jsonl(out / "viewpoints_t1.jsonl", current_viewpoints)
        write_json(out / "change_log.json", change_log)
        report_path = out / "report_t1.md"
        report_path.write_text(render_t1_report(case_id=case_id, from_snapshot=from_snapshot, to_snapshot=to_snapshot, current_viewpoints=current_viewpoints, event=event, change_log=change_log), encoding="utf-8")
        approval(
            store,
            root,
            case_id=case_id,
            gate="G4_T1_REPORT",
            artifact=str(report_path.relative_to(root)),
            approval_file=report_approval_file,
            run_id=actual_run,
        )
        store.record_model_run(actual_run, case_id, "T1", "manual_synthetic", {"llm_called": False, "updated": approved_affected})
        manifest = save_manifest(
            root,
            actual_run,
            case_id,
            stage="A",
            artifacts={
                "delta_evidence": str((out / "delta_evidence_t1.jsonl").relative_to(root)),
                "affected_viewpoints": str((out / "affected_viewpoints.json").relative_to(root)),
                "viewpoints_t1": str((out / "viewpoints_t1.jsonl").relative_to(root)),
                "change_log": str((out / "change_log.json").relative_to(root)),
                "report_t1": str(report_path.relative_to(root)),
            },
            snapshots=[from_snapshot_id, to_snapshot_id],
            extra={"candidate_count": len(candidates), "updated_count": len(new_viewpoints), "unchanged_count": len(unchanged), "gate": "G4_T1"},
        )
    return {"case_id": case_id, "run_id": actual_run, "updated": [f"{v.viewpoint_id}:{v.version}" for v in new_viewpoints], "unchanged": unchanged, "manifest": str(manifest)}


def run_synthetic(workspace: str | Path) -> Dict[str, Any]:
    root = workspace_path(workspace)
    run_id = new_run_id("phase-a-demo")
    config = root / "config" / "case_synthetic.yaml"
    init_case(root, config, run_id=run_id)
    ingest(root, case_id="synthetic_military_ai", phase=SnapshotTag.T0, input_path=root / "data" / "raw" / "t0", run_id=run_id)
    t0 = freeze_snapshot(root, case_id="synthetic_military_ai", phase=SnapshotTag.T0, approval_file=root / "fixtures" / "approvals" / "g1_t0.json", run_id=run_id)
    freeze_hypotheses(root, case_id="synthetic_military_ai", snapshot_id=t0["snapshot_id"], input_path=root / "fixtures" / "hypotheses_t0.json", approval_file=root / "fixtures" / "approvals" / "g2.json", run_id=run_id)
    analyze_t0(root, case_id="synthetic_military_ai", snapshot_id=t0["snapshot_id"], input_dir=root / "fixtures" / "analysis_t0", monitor_approval_file=root / "fixtures" / "approvals" / "g3.json", report_approval_file=root / "fixtures" / "approvals" / "g4_t0.json", run_id=run_id)
    ingest(root, case_id="synthetic_military_ai", phase=SnapshotTag.T1, input_path=root / "data" / "raw" / "t1", run_id=run_id)
    t1 = freeze_snapshot(root, case_id="synthetic_military_ai", phase=SnapshotTag.T1, approval_file=root / "fixtures" / "approvals" / "g1_t1.json", run_id=run_id)
    result = update_t1(root, case_id="synthetic_military_ai", from_snapshot_id=t0["snapshot_id"], to_snapshot_id=t1["snapshot_id"], input_path=root / "fixtures" / "update_t1.json", affected_approval_file=root / "fixtures" / "approvals" / "affected_t1.json", report_approval_file=root / "fixtures" / "approvals" / "g4_t1.json", run_id=run_id)
    result.update({"t0_snapshot": t0["snapshot_id"], "t1_snapshot": t1["snapshot_id"], "run_id": run_id})
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase A no-LLM research prototype")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-case")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id")

    p = sub.add_parser("ingest")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--phase", required=True, choices=["t0", "t1"])
    p.add_argument("--input", required=True, dest="input_path")
    p.add_argument("--run-id")

    p = sub.add_parser("freeze")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--phase", required=True, choices=["t0", "t1"])
    p.add_argument("--approve-file")
    p.add_argument("--cutoff")
    p.add_argument("--run-id")

    p = sub.add_parser("hypotheses")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--snapshot", required=True, dest="snapshot_id")
    p.add_argument("--input", required=True, dest="input_path")
    p.add_argument("--approve-file")
    p.add_argument("--run-id")

    p = sub.add_parser("analyze-t0")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--snapshot", required=True, dest="snapshot_id")
    p.add_argument("--input-dir", required=True, dest="input_dir")
    p.add_argument("--approve-monitor-file")
    p.add_argument("--approve-report-file")
    p.add_argument("--run-id")

    p = sub.add_parser("build-monitor-plan")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--snapshot", required=True, dest="snapshot_id")
    p.add_argument("--run-id")

    p = sub.add_parser("update-t1")
    p.add_argument("--case", required=True, dest="case_id")
    p.add_argument("--from", required=True, dest="from_snapshot_id")
    p.add_argument("--to", required=True, dest="to_snapshot_id")
    p.add_argument("--input", required=True, dest="input_path")
    p.add_argument("--approve-affected-file")
    p.add_argument("--approve-report-file")
    p.add_argument("--run-id")

    p = sub.add_parser("run-synthetic")
    p.add_argument("--run-id")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-case":
            result = init_case(args.workspace, args.config, args.run_id)
        elif args.command == "ingest":
            result = ingest(args.workspace, case_id=args.case_id, phase=SnapshotTag(args.phase.upper()), input_path=args.input_path, run_id=args.run_id)
        elif args.command == "freeze":
            result = freeze_snapshot(args.workspace, case_id=args.case_id, phase=SnapshotTag(args.phase.upper()), approval_file=args.approve_file, cutoff=args.cutoff, run_id=args.run_id)
        elif args.command == "hypotheses":
            result = freeze_hypotheses(args.workspace, case_id=args.case_id, snapshot_id=args.snapshot_id, input_path=args.input_path, approval_file=args.approve_file, run_id=args.run_id)
        elif args.command == "analyze-t0":
            result = analyze_t0(args.workspace, case_id=args.case_id, snapshot_id=args.snapshot_id, input_dir=args.input_dir, monitor_approval_file=args.approve_monitor_file, report_approval_file=args.approve_report_file, run_id=args.run_id)
        elif args.command == "build-monitor-plan":
            result = build_monitor_plan(args.workspace, case_id=args.case_id, snapshot_id=args.snapshot_id, run_id=args.run_id)
        elif args.command == "update-t1":
            result = update_t1(args.workspace, case_id=args.case_id, from_snapshot_id=args.from_snapshot_id, to_snapshot_id=args.to_snapshot_id, input_path=args.input_path, affected_approval_file=args.approve_affected_file, report_approval_file=args.approve_report_file, run_id=args.run_id)
        elif args.command == "run-synthetic":
            result = run_synthetic(args.workspace)
        else:  # pragma: no cover
            raise ValueError(f"Unknown command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except GatePending as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (StorageError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
