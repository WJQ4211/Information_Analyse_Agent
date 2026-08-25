"""SQLite persistence for append-only Phase A objects."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type

from .schemas import (
    DiscriminativeNeed,
    Disagreement,
    Evidence,
    EvidenceDependency,
    EvidenceSnapshot,
    EvidenceStatus,
    Hypothesis,
    IndicatorStatus,
    MonitorIndicator,
    SnapshotTag,
    TriggerRule,
    UpdateEvent,
    Viewpoint,
    ViewpointStatus,
    evidence_ref,
    model_dump,
    model_validate,
    parse_evidence_ref,
    stable_hash,
    with_evidence_hash,
)


class StorageError(RuntimeError):
    """Raised when a business or append-only storage constraint is violated."""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migration (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_config (
    case_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version >= 1),
    snapshot_tag TEXT NOT NULL CHECK(snapshot_tag IN ('T0', 'T1')),
    status TEXT NOT NULL,
    published_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (evidence_id, version)
);

CREATE TABLE IF NOT EXISTS evidence_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK(phase IN ('T0', 'T1')),
    cutoff_time TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    frozen INTEGER NOT NULL CHECK(frozen IN (0, 1)),
    frozen_at TEXT NOT NULL,
    frozen_by TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshot_items (
    snapshot_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY(snapshot_id, evidence_id, version),
    FOREIGN KEY(snapshot_id) REFERENCES evidence_snapshot(snapshot_id),
    FOREIGN KEY(evidence_id, version) REFERENCES evidence(evidence_id, version)
);

CREATE TABLE IF NOT EXISTS hypothesis (
    hypothesis_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    frozen INTEGER NOT NULL CHECK(frozen IN (0, 1)),
    payload_json TEXT NOT NULL,
    FOREIGN KEY(snapshot_id) REFERENCES evidence_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS viewpoint (
    viewpoint_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version >= 1),
    snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(viewpoint_id, version),
    FOREIGN KEY(snapshot_id) REFERENCES evidence_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS evidence_dependency (
    edge_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    evidence_version INTEGER NOT NULL,
    viewpoint_id TEXT NOT NULL,
    viewpoint_version INTEGER NOT NULL,
    relation TEXT NOT NULL,
    importance TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(evidence_id, evidence_version) REFERENCES evidence(evidence_id, version),
    FOREIGN KEY(viewpoint_id, viewpoint_version) REFERENCES viewpoint(viewpoint_id, version)
);

CREATE TABLE IF NOT EXISTS disagreement (
    disagreement_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    decision_impact TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(snapshot_id) REFERENCES evidence_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS discriminative_need (
    need_id TEXT PRIMARY KEY,
    disagreement_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(disagreement_id) REFERENCES disagreement(disagreement_id)
);

CREATE TABLE IF NOT EXISTS monitor_indicator (
    indicator_id TEXT PRIMARY KEY,
    need_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(need_id) REFERENCES discriminative_need(need_id)
);

CREATE TABLE IF NOT EXISTS trigger_rule (
    trigger_id TEXT PRIMARY KEY,
    indicator_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(indicator_id) REFERENCES monitor_indicator(indicator_id)
);

CREATE TABLE IF NOT EXISTS update_event (
    update_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    from_snapshot TEXT NOT NULL,
    to_snapshot TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(from_snapshot) REFERENCES evidence_snapshot(snapshot_id),
    FOREIGN KEY(to_snapshot) REFERENCES evidence_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS model_run (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    generator_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_review (
    review_id TEXT PRIMARY KEY,
    gate TEXT NOT NULL,
    case_id TEXT NOT NULL,
    approved INTEGER NOT NULL CHECK(approved IN (0, 1)),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact (
    artifact_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS prevent_frozen_snapshot_update
BEFORE UPDATE ON evidence_snapshot
WHEN OLD.frozen = 1
BEGIN
    SELECT RAISE(ABORT, 'frozen evidence snapshot is immutable');
END;

CREATE TRIGGER IF NOT EXISTS prevent_frozen_snapshot_delete
BEFORE DELETE ON evidence_snapshot
WHEN OLD.frozen = 1
BEGIN
    SELECT RAISE(ABORT, 'frozen evidence snapshot is immutable');
END;

CREATE TRIGGER IF NOT EXISTS prevent_frozen_snapshot_item_update
BEFORE UPDATE ON snapshot_items
WHEN (SELECT frozen FROM evidence_snapshot WHERE snapshot_id = OLD.snapshot_id) = 1
BEGIN
    SELECT RAISE(ABORT, 'items of a frozen evidence snapshot are immutable');
END;

CREATE TRIGGER IF NOT EXISTS prevent_frozen_snapshot_item_delete
BEFORE DELETE ON snapshot_items
WHEN (SELECT frozen FROM evidence_snapshot WHERE snapshot_id = OLD.snapshot_id) = 1
BEGIN
    SELECT RAISE(ABORT, 'items of a frozen evidence snapshot are immutable');
END;
"""


def _utc_text(value: datetime) -> str:
    return value.isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_payload(row: sqlite3.Row) -> Dict[str, Any]:
    return json.loads(row["payload_json"])


def _canonical_payload(value: Any) -> Any:
    """Ignore runtime metadata when checking idempotent fixture replays."""
    if isinstance(value, dict):
        return {
            key: _canonical_payload(item)
            for key, item in value.items()
            if key not in {"created_at", "run_id", "created_by"}
        }
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    return value


class Store:
    """Small repository enforcing the Phase A append-only rules."""

    def __init__(self, path: str | Path = "data/research.sqlite3") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migration(version, applied_at) VALUES (?, ?)",
            (1, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def save_case_config(self, case_id: str, config: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO case_config(case_id, config_json, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(case_id) DO UPDATE SET config_json=excluded.config_json",
            (case_id, _json(config), datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def get_case_config(self, case_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT config_json FROM case_config WHERE case_id = ?", (case_id,)
        ).fetchone()
        return json.loads(row["config_json"]) if row else None

    def record_model_run(
        self, run_id: str, case_id: str, phase: str, generator_type: str, payload: Dict[str, Any]
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO model_run(run_id, case_id, phase, generator_type, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, case_id, phase, generator_type, _json(payload)),
        )
        self.conn.commit()

    def record_human_review(self, review: Any) -> None:
        data = model_dump(review)
        self.conn.execute(
            "INSERT OR REPLACE INTO human_review(review_id, gate, case_id, approved, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (data["review_id"], data["gate"], data["case_id"], int(data["approved"]), _json(data)),
        )
        self.conn.commit()

    def insert_evidence(self, evidence: Evidence) -> Evidence:
        evidence = with_evidence_hash(evidence)
        data = model_dump(evidence)
        row = self.conn.execute(
            "SELECT payload_json FROM evidence WHERE evidence_id=? AND version=?",
            (evidence.evidence_id, evidence.version),
        ).fetchone()
        if row:
            existing_data = json.loads(row["payload_json"])
            comparable_existing = dict(existing_data)
            comparable_data = dict(data)
            for field in ("created_at", "run_id", "schema_version", "content_hash"):
                comparable_existing.pop(field, None)
                comparable_data.pop(field, None)
            if _canonical_payload(comparable_existing) != _canonical_payload(comparable_data):
                raise StorageError(
                    f"Evidence {evidence_ref(evidence.evidence_id, evidence.version)} already exists with different content"
                )
            return model_validate(Evidence, existing_data)  # type: ignore[return-value]
        self.conn.execute(
            "INSERT INTO evidence(evidence_id, version, snapshot_tag, status, published_at, content_hash, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                evidence.evidence_id,
                evidence.version,
                evidence.snapshot_tag.value,
                evidence.status.value,
                evidence.published_at.isoformat(),
                evidence.content_hash,
                _json(data),
            ),
        )
        self.conn.commit()
        return evidence

    def get_evidence(self, evidence_id: str, version: int) -> Optional[Evidence]:
        row = self.conn.execute(
            "SELECT payload_json FROM evidence WHERE evidence_id=? AND version=?",
            (evidence_id, version),
        ).fetchone()
        return model_validate(Evidence, _read_payload(row)) if row else None  # type: ignore[return-value]

    def list_evidence(self, *, tags: Optional[Sequence[SnapshotTag]] = None) -> List[Evidence]:
        if tags:
            placeholders = ",".join("?" for _ in tags)
            rows = self.conn.execute(
                f"SELECT payload_json FROM evidence WHERE snapshot_tag IN ({placeholders}) "
                "ORDER BY published_at, evidence_id, version",
                [tag.value for tag in tags],
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM evidence ORDER BY published_at, evidence_id, version"
            ).fetchall()
        return [model_validate(Evidence, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def create_snapshot(
        self,
        *,
        snapshot_id: str,
        case_id: str,
        phase: SnapshotTag,
        cutoff_time: datetime,
        evidence_versions: Iterable[str],
        frozen_by: str,
        run_id: str,
    ) -> EvidenceSnapshot:
        refs = sorted(set(evidence_versions))
        if not refs:
            raise StorageError("Cannot freeze an empty evidence snapshot")
        existing = self.get_snapshot(snapshot_id)
        if existing:
            if existing.evidence_versions != refs or existing.manifest_hash != stable_hash(refs):
                raise StorageError(f"Snapshot {snapshot_id} already exists with different content")
            return existing
        evidence_objects: List[Evidence] = []
        for ref in refs:
            evidence_id, version = parse_evidence_ref(ref)
            evidence = self.get_evidence(evidence_id, version)
            if evidence is None:
                raise StorageError(f"Snapshot references missing evidence: {ref}")
            if phase == SnapshotTag.T0 and evidence.snapshot_tag != SnapshotTag.T0:
                raise StorageError(f"T0 snapshot cannot contain {evidence.snapshot_tag.value} evidence: {ref}")
            if evidence.published_at.isoformat() > cutoff_time.date().isoformat():
                raise StorageError(f"Evidence is newer than snapshot cutoff: {ref}")
            evidence_objects.append(evidence)
        if phase == SnapshotTag.T1 and not any(e.snapshot_tag == SnapshotTag.T1 for e in evidence_objects):
            raise StorageError("T1 snapshot must contain at least one T1 evidence item")
        manifest_hash = stable_hash(refs)
        snapshot = EvidenceSnapshot(
            snapshot_id=snapshot_id,
            case_id=case_id,
            phase=phase,
            cutoff_time=cutoff_time,
            evidence_versions=refs,
            manifest_hash=manifest_hash,
            frozen=True,
            frozen_by=frozen_by,
            run_id=run_id,
        )
        data = model_dump(snapshot)
        with self.transaction():
            self.conn.execute(
                "INSERT INTO evidence_snapshot(snapshot_id, case_id, phase, cutoff_time, manifest_hash, frozen, frozen_at, frozen_by, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.snapshot_id,
                    snapshot.case_id,
                    snapshot.phase.value,
                    snapshot.cutoff_time.isoformat(),
                    snapshot.manifest_hash,
                    int(snapshot.frozen),
                    snapshot.frozen_at.isoformat(),
                    snapshot.frozen_by,
                    _json(data),
                ),
            )
            for ref in refs:
                evidence_id, version = parse_evidence_ref(ref)
                self.conn.execute(
                    "INSERT INTO snapshot_items(snapshot_id, evidence_id, version) VALUES (?, ?, ?)",
                    (snapshot.snapshot_id, evidence_id, version),
                )
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> Optional[EvidenceSnapshot]:
        row = self.conn.execute(
            "SELECT payload_json FROM evidence_snapshot WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        return model_validate(EvidenceSnapshot, _read_payload(row)) if row else None  # type: ignore[return-value]

    def list_snapshots(self, case_id: Optional[str] = None) -> List[EvidenceSnapshot]:
        if case_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM evidence_snapshot WHERE case_id=? ORDER BY frozen_at",
                (case_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM evidence_snapshot ORDER BY frozen_at"
            ).fetchall()
        return [model_validate(EvidenceSnapshot, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_hypotheses(self, hypotheses: Iterable[Hypothesis]) -> None:
        with self.transaction():
            for hypothesis in hypotheses:
                snapshot = self.get_snapshot(hypothesis.snapshot_id)
                if not snapshot or not snapshot.frozen:
                    raise StorageError("Hypotheses require a frozen snapshot")
                if not hypothesis.frozen:
                    raise StorageError(f"Hypothesis {hypothesis.hypothesis_id} is not frozen at G2")
                if not hypothesis.observable_implications or not hypothesis.falsifiers:
                    raise StorageError(f"Hypothesis {hypothesis.hypothesis_id} needs observable implications and falsifiers")
                data = model_dump(hypothesis)
                row = self.conn.execute(
                    "SELECT payload_json FROM hypothesis WHERE hypothesis_id=?",
                    (hypothesis.hypothesis_id,),
                ).fetchone()
                if row:
                    if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                        raise StorageError(f"Hypothesis {hypothesis.hypothesis_id} already exists with different content")
                    continue
                self.conn.execute(
                    "INSERT INTO hypothesis(hypothesis_id, snapshot_id, frozen, payload_json) VALUES (?, ?, ?, ?)",
                    (hypothesis.hypothesis_id, hypothesis.snapshot_id, int(hypothesis.frozen), _json(data)),
                )

    def list_hypotheses(self, snapshot_id: str) -> List[Hypothesis]:
        rows = self.conn.execute(
            "SELECT payload_json FROM hypothesis WHERE snapshot_id=? ORDER BY hypothesis_id",
            (snapshot_id,),
        ).fetchall()
        return [model_validate(Hypothesis, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def snapshot_contains(self, snapshot_id: str, evidence_id: str, version: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM snapshot_items WHERE snapshot_id=? AND evidence_id=? AND version=?",
            (snapshot_id, evidence_id, version),
        ).fetchone()
        return row is not None

    def insert_viewpoint(self, viewpoint: Viewpoint) -> None:
        snapshot = self.get_snapshot(viewpoint.snapshot_id)
        if not snapshot or not snapshot.frozen:
            raise StorageError("Viewpoints require a frozen evidence snapshot")
        all_citations = viewpoint.supporting_evidence + viewpoint.counter_evidence
        if viewpoint.status in {ViewpointStatus.ADMITTED, ViewpointStatus.CONDITIONAL} and not viewpoint.supporting_evidence:
            raise StorageError(f"Admitted viewpoint {viewpoint.viewpoint_id} must cite supporting evidence")
        if viewpoint.status in {ViewpointStatus.ADMITTED, ViewpointStatus.CONDITIONAL}:
            required_lists = {
                "key_assumptions": viewpoint.key_assumptions,
                "mechanism_claims": viewpoint.mechanism_claims,
                "boundary_conditions": viewpoint.boundary_conditions,
                "falsifiers": viewpoint.falsifiers,
                "uncertainties": viewpoint.uncertainties,
            }
            missing = [name for name, values in required_lists.items() if not values]
            if missing:
                raise StorageError(f"Admitted viewpoint {viewpoint.viewpoint_id} has empty required fields: {missing}")
        for citation in all_citations:
            if not self.snapshot_contains(viewpoint.snapshot_id, citation.evidence_id, citation.version):
                raise StorageError(
                    f"Viewpoint {viewpoint.viewpoint_id} cites evidence outside snapshot: {citation.ref}"
                )
        data = model_dump(viewpoint)
        existing_row = self.conn.execute(
            "SELECT payload_json FROM viewpoint WHERE viewpoint_id=? AND version=?",
            (viewpoint.viewpoint_id, viewpoint.version),
        ).fetchone()
        if existing_row:
            if _canonical_payload(json.loads(existing_row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Viewpoint {viewpoint.viewpoint_id} v{viewpoint.version} already exists with different content")
            return
        latest_row = self.conn.execute(
            "SELECT MAX(version) AS max_version FROM viewpoint WHERE viewpoint_id=?",
            (viewpoint.viewpoint_id,),
        ).fetchone()
        latest = latest_row["max_version"]
        if viewpoint.version == 1:
            if viewpoint.parent_version is not None:
                raise StorageError("Initial viewpoint cannot have parent_version")
        else:
            if latest is None or viewpoint.version != latest + 1:
                raise StorageError("Viewpoint versions must append exactly one version")
            if viewpoint.parent_version != latest:
                raise StorageError("New viewpoint version must reference the previous version")
            if not viewpoint.change_reason.strip() or viewpoint.change_reason == "initial_version":
                raise StorageError("New viewpoint version requires change_reason")
        self.conn.execute(
            "INSERT INTO viewpoint(viewpoint_id, version, snapshot_id, status, payload_json) VALUES (?, ?, ?, ?, ?)",
            (viewpoint.viewpoint_id, viewpoint.version, viewpoint.snapshot_id, viewpoint.status.value, _json(data)),
        )
        self.conn.commit()

    def get_viewpoint(self, viewpoint_id: str, version: int) -> Optional[Viewpoint]:
        row = self.conn.execute(
            "SELECT payload_json FROM viewpoint WHERE viewpoint_id=? AND version=?",
            (viewpoint_id, version),
        ).fetchone()
        return model_validate(Viewpoint, _read_payload(row)) if row else None  # type: ignore[return-value]

    def latest_viewpoint(self, viewpoint_id: str) -> Optional[Viewpoint]:
        row = self.conn.execute(
            "SELECT payload_json FROM viewpoint WHERE viewpoint_id=? ORDER BY version DESC LIMIT 1",
            (viewpoint_id,),
        ).fetchone()
        return model_validate(Viewpoint, _read_payload(row)) if row else None  # type: ignore[return-value]

    def list_viewpoints(self, snapshot_id: Optional[str] = None) -> List[Viewpoint]:
        if snapshot_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM viewpoint WHERE snapshot_id=? ORDER BY viewpoint_id, version",
                (snapshot_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM viewpoint ORDER BY viewpoint_id, version"
            ).fetchall()
        return [model_validate(Viewpoint, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_dependency(self, dependency: EvidenceDependency) -> None:
        evidence = self.get_evidence(dependency.evidence_id, dependency.evidence_version)
        viewpoint = self.get_viewpoint(dependency.viewpoint_id, dependency.viewpoint_version)
        if not evidence or not viewpoint:
            raise StorageError("Dependency must reference existing evidence and viewpoint versions")
        if not self.snapshot_contains(viewpoint.snapshot_id, dependency.evidence_id, dependency.evidence_version):
            raise StorageError("Dependency evidence must be present in the viewpoint snapshot")
        data = model_dump(dependency)
        row = self.conn.execute(
            "SELECT payload_json FROM evidence_dependency WHERE edge_id=?", (dependency.edge_id,)
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Dependency {dependency.edge_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO evidence_dependency(edge_id, evidence_id, evidence_version, viewpoint_id, viewpoint_version, relation, importance, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dependency.edge_id,
                dependency.evidence_id,
                dependency.evidence_version,
                dependency.viewpoint_id,
                dependency.viewpoint_version,
                dependency.relation.value,
                dependency.importance.value,
                _json(data),
            ),
        )
        self.conn.commit()

    def list_dependencies(self, *, viewpoint_id: Optional[str] = None) -> List[EvidenceDependency]:
        if viewpoint_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM evidence_dependency WHERE viewpoint_id=? ORDER BY edge_id",
                (viewpoint_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM evidence_dependency ORDER BY edge_id"
            ).fetchall()
        return [model_validate(EvidenceDependency, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_disagreement(self, disagreement: Disagreement) -> None:
        if not self.get_snapshot(disagreement.snapshot_id):
            raise StorageError("Disagreement must reference an existing snapshot")
        if len(disagreement.viewpoint_refs) < 2:
            raise StorageError("Disagreement must connect at least two viewpoint versions")
        for ref in disagreement.viewpoint_refs:
            if ":" not in ref:
                raise StorageError(f"Invalid viewpoint reference: {ref}")
            vp_id, version_text = ref.rsplit(":", 1)
            if not self.get_viewpoint(vp_id, int(version_text)):
                raise StorageError(f"Disagreement references missing viewpoint: {ref}")
        data = model_dump(disagreement)
        row = self.conn.execute(
            "SELECT payload_json FROM disagreement WHERE disagreement_id=?",
            (disagreement.disagreement_id,),
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Disagreement {disagreement.disagreement_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO disagreement(disagreement_id, snapshot_id, decision_impact, status, payload_json) VALUES (?, ?, ?, ?, ?)",
            (
                disagreement.disagreement_id,
                disagreement.snapshot_id,
                disagreement.decision_impact.value,
                disagreement.status.value,
                _json(data),
            ),
        )
        self.conn.commit()

    def list_disagreements(self, snapshot_id: str) -> List[Disagreement]:
        rows = self.conn.execute(
            "SELECT payload_json FROM disagreement WHERE snapshot_id=? ORDER BY disagreement_id",
            (snapshot_id,),
        ).fetchall()
        return [model_validate(Disagreement, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_need(self, need: DiscriminativeNeed) -> None:
        if not self.conn.execute(
            "SELECT 1 FROM disagreement WHERE disagreement_id=?", (need.disagreement_id,)
        ).fetchone():
            raise StorageError(f"Need references missing disagreement: {need.disagreement_id}")
        data = model_dump(need)
        row = self.conn.execute(
            "SELECT payload_json FROM discriminative_need WHERE need_id=?", (need.need_id,)
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Need {need.need_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO discriminative_need(need_id, disagreement_id, payload_json) VALUES (?, ?, ?)",
            (need.need_id, need.disagreement_id, _json(data)),
        )
        self.conn.commit()

    def list_needs(self, disagreement_id: Optional[str] = None) -> List[DiscriminativeNeed]:
        if disagreement_id:
            rows = self.conn.execute(
                "SELECT payload_json FROM discriminative_need WHERE disagreement_id=? ORDER BY need_id",
                (disagreement_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT payload_json FROM discriminative_need ORDER BY need_id"
            ).fetchall()
        return [model_validate(DiscriminativeNeed, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_indicator(self, indicator: MonitorIndicator) -> None:
        if not self.conn.execute(
            "SELECT 1 FROM discriminative_need WHERE need_id=?", (indicator.need_id,)
        ).fetchone():
            raise StorageError(f"Indicator references missing need: {indicator.need_id}")
        data = model_dump(indicator)
        row = self.conn.execute(
            "SELECT payload_json FROM monitor_indicator WHERE indicator_id=?", (indicator.indicator_id,)
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Indicator {indicator.indicator_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO monitor_indicator(indicator_id, need_id, status, payload_json) VALUES (?, ?, ?, ?)",
            (indicator.indicator_id, indicator.need_id, indicator.status.value, _json(data)),
        )
        self.conn.commit()

    def list_indicators(self, *, approved_only: bool = False) -> List[MonitorIndicator]:
        query = "SELECT payload_json FROM monitor_indicator"
        params: Tuple[Any, ...] = ()
        if approved_only:
            query += " WHERE status=?"
            params = (IndicatorStatus.APPROVED.value,)
        query += " ORDER BY indicator_id"
        rows = self.conn.execute(query, params).fetchall()
        return [model_validate(MonitorIndicator, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_trigger(self, trigger: TriggerRule) -> None:
        if not self.conn.execute(
            "SELECT 1 FROM monitor_indicator WHERE indicator_id=?", (trigger.indicator_id,)
        ).fetchone():
            raise StorageError(f"Trigger references missing indicator: {trigger.indicator_id}")
        data = model_dump(trigger)
        row = self.conn.execute(
            "SELECT payload_json FROM trigger_rule WHERE trigger_id=?", (trigger.trigger_id,)
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Trigger {trigger.trigger_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO trigger_rule(trigger_id, indicator_id, status, payload_json) VALUES (?, ?, ?, ?)",
            (trigger.trigger_id, trigger.indicator_id, trigger.status.value, _json(data)),
        )
        self.conn.commit()

    def list_triggers(self, *, approved_only: bool = False) -> List[TriggerRule]:
        query = "SELECT payload_json FROM trigger_rule"
        params: Tuple[Any, ...] = ()
        if approved_only:
            query += " WHERE status=?"
            params = (IndicatorStatus.APPROVED.value,)
        query += " ORDER BY trigger_id"
        rows = self.conn.execute(query, params).fetchall()
        return [model_validate(TriggerRule, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def insert_update_event(self, event: UpdateEvent) -> None:
        data = model_dump(event)
        row = self.conn.execute(
            "SELECT payload_json FROM update_event WHERE update_id=?", (event.update_id,)
        ).fetchone()
        if row:
            if _canonical_payload(json.loads(row["payload_json"])) != _canonical_payload(data):
                raise StorageError(f"Update event {event.update_id} already exists with different content")
            return
        self.conn.execute(
            "INSERT INTO update_event(update_id, case_id, from_snapshot, to_snapshot, payload_json) VALUES (?, ?, ?, ?, ?)",
            (event.update_id, event.case_id, event.from_snapshot, event.to_snapshot, _json(data)),
        )
        self.conn.commit()

    def list_update_events(self, case_id: str) -> List[UpdateEvent]:
        rows = self.conn.execute(
            "SELECT payload_json FROM update_event WHERE case_id=? ORDER BY update_id", (case_id,)
        ).fetchall()
        return [model_validate(UpdateEvent, _read_payload(row)) for row in rows]  # type: ignore[list-item]

    def record_artifact(
        self, *, artifact_id: str, case_id: str, artifact_type: str, path: str, content_hash: str, payload: Dict[str, Any]
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO artifact(artifact_id, case_id, artifact_type, path, content_hash, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (artifact_id, case_id, artifact_type, path, content_hash, _json(payload)),
        )
        self.conn.commit()
