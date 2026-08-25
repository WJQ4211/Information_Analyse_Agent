"""Run the bounded P3a evidence-extraction pilot for two selected T0 sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_client import FileBridgeModelClient
from src.p3a import (
    LocatorText,
    candidate_payload,
    inject_deterministic_metadata,
    normalize_for_match,
    validate_candidate,
)
from src.schemas import EvidenceCandidate, model_dump


ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "us_military_ai_deployment"
P2_RUN_ID = "P2-20260825T154432Z-author-pilot"
P2_MANIFEST = ROOT / "outputs" / P2_RUN_ID / "p2_manifest.json"
P2_SOURCE = ROOT / "data/curated/source_manifest_t0_p2.csv"
P2_INVENTORY = ROOT / "outputs" / P2_RUN_ID / "selected_source_inventory.csv"
P1_DIR = ROOT / "outputs/P1-20260825T081142Z-7f4c9b"
P1B_DIR = ROOT / "outputs/P1B-20260825T140911Z-manual"
P1C_DIR = ROOT / "outputs/P1C-20260825T152730Z-manual-web"
PROMPT_TEMPLATE = ROOT / "prompts/evidence_extract_v1.yaml"
SP_PAGE_INDEX = P1B_DIR / "page_index/T0-SP-007.jsonl"
TD_TEXT = P1C_DIR / "web_text/T0-TD-008.txt"
SOURCE_IDS = ["T0-SP-007", "T0-TD-008"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def current_git_head() -> dict[str, str]:
    for candidate in (ROOT / ".github_upload_staging", ROOT):
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        value = result.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return {"commit": value, "short_commit": value[:12], "git_root": str(candidate)}
    raise RuntimeError("A real Git HEAD is required for P3a; refusing unverified_local_copy.")


def protected_paths() -> list[Path]:
    paths = [
        ROOT / "data/curated/source_manifest_t0.csv",
        ROOT / "data/curated/search_log_t0.csv",
        ROOT / "data/curated/source_manifest_t0_p1b.csv",
        ROOT / "data/curated/search_log_t0_p1b.csv",
        ROOT / "data/curated/source_manifest_t0_p1c.csv",
        ROOT / "data/curated/p2_review_template.csv",
        ROOT / "data/curated/p2_review_decisions_t0.csv",
        P2_SOURCE,
        P2_MANIFEST,
        P2_INVENTORY,
    ]
    for directory in (P1_DIR, P1B_DIR, P1C_DIR, ROOT / "outputs" / P2_RUN_ID):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(set(path for path in paths if path.is_file()))


def hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def load_source_metadata() -> dict[str, dict[str, str]]:
    rows = {row["source_id"]: row for row in read_csv(P2_SOURCE)}
    if set(SOURCE_IDS) - set(rows):
        raise RuntimeError("P2 source manifest is missing a P3a source")
    return {source_id: {**rows[source_id], "case_id": CASE_ID} for source_id in SOURCE_IDS}


def load_sp_pages() -> dict[str, LocatorText]:
    pages: dict[str, LocatorText] = {}
    for line in SP_PAGE_INDEX.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        locator = f"pdf_page_{row['pdf_page']}"
        pages[locator] = LocatorText("T0-SP-007", locator, row["text_raw"])
    if not pages:
        raise RuntimeError("SP-007 page index is empty")
    return pages


def load_td_paragraphs() -> dict[str, LocatorText]:
    text = TD_TEXT.read_text(encoding="utf-8")
    start = text.find("MANAMA, Bahrain -")
    end = text.find("Share PRINT RSS Press Briefings")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("TD-008 article boundaries were not found in the P1c body text")
    article = normalize_for_match(text[start:end])
    article = re.sub(
        r"Download Download Share PRINT RSS Tags Unmanned Systems Exercise U\.S\. 5th Fleet AI\s*",
        "",
        article,
    )
    markers = [
        r"MANAMA, Bahrain -",
        r"IMSC[^\w\s]s operational task force",
        r"[\"“\ufffd]+We planned this exercise",
        r"During the exercise",
        r"[\"“\ufffd]+Saildrones transmitted",
        r"CTF Sentinel previously",
        r"The late-summer event",
        r"IMSC was formed",
        r"IMSC[^\w\s]s operational arm",
        r"IMSC membership currently",
    ]
    positions: list[int] = []
    for marker in markers:
        match = re.search(marker, article)
        if match and match.start() not in positions:
            positions.append(match.start())
    positions.sort()
    paragraphs: dict[str, LocatorText] = {}
    for index, position in enumerate(positions):
        next_position = positions[index + 1] if index + 1 < len(positions) else len(article)
        value = article[position:next_position].strip()
        locator = f"td008/article/paragraph-{index + 1:02d}"
        paragraphs[locator] = LocatorText("T0-TD-008", locator, value)
    if len(paragraphs) < 8:
        raise RuntimeError(f"TD-008 stable paragraph split produced only {len(paragraphs)} paragraphs")
    return paragraphs


def extract_span(text: str, start: str, end: str) -> str:
    normalized = normalize_for_match(text)
    start_value = normalize_for_match(start)
    end_value = normalize_for_match(end)
    lower = normalized.casefold()
    start_index = lower.find(start_value.casefold())
    if start_index < 0:
        raise RuntimeError(f"Could not find excerpt start anchor: {start}")
    end_index = lower.find(end_value.casefold(), start_index + len(start_value))
    if end_index < 0:
        raise RuntimeError(f"Could not find excerpt end anchor: {end}")
    return normalized[start_index:end_index + len(end_value)]


def candidate_payloads(sp_pages: dict[str, LocatorText], td_paragraphs: dict[str, LocatorText]) -> dict[str, list[dict[str, Any]]]:
    def sp(page: int, start: str, end: str, claim: str, dimensions: list[str], nature: str, topics: list[str]) -> dict[str, Any]:
        locator = f"pdf_page_{page}"
        excerpt = extract_span(sp_pages[locator].text, start, end)
        return {
            "source_id": "T0-SP-007",
            "source_locator": locator,
            "excerpt_original": excerpt,
            "excerpt_zh": "",
            "normalized_claim": claim,
            "coding_dimensions": dimensions,
            "evidence_nature": nature,
            "topics": topics,
        }

    def td(locator: str, claim: str, dimensions: list[str], nature: str, topics: list[str], extra_locators: list[str] | None = None) -> dict[str, Any]:
        locators = [locator] + (extra_locators or [])
        excerpt = " ".join(td_paragraphs[item].text for item in locators)
        actual_locator = locator if not extra_locators else f"{locator}--{extra_locators[-1].split('-')[-1]}"
        return {
            "source_id": "T0-TD-008",
            "source_locator": actual_locator,
            "excerpt_original": excerpt,
            "excerpt_zh": "",
            "normalized_claim": claim,
            "coding_dimensions": dimensions,
            "evidence_nature": nature,
            "topics": topics,
        }

    return {
        "T0-SP-007-batch-01": [
            sp(4, "Modern ize gove rnance", "technology will be used;", "The strategy sets continuous oversight of DoD AI use as a governance goal that should account for the context of use.", ["trust_governance_constraints"], "goal", ["governance", "oversight"]),
            sp(4, "Exercise approp riate care", "National Defense Stra tegy:", "The strategy calls for considering AI risks from the outset of a project and mitigating unintended consequences while enabling development.", ["trust_governance_constraints", "application_maturity"], "goal", ["acquisition_lifecycle", "risk_management"]),
            sp(7, "RELIABLE:", "life-cycle s.", "The stated reliable-AI principle subjects safety, security, and effectiveness to testing and assurance within defined uses across the life cycle.", ["trust_governance_constraints"], "goal", ["reliability", "testing"]),
            sp(12, "Use the requirements validation", "prior to and during deplo yment.", "The requirements-validation goal links AI capabilities to operational needs and relevant risks, and describes validation as increasing reliability and safety before and during deployment.", ["trust_governance_constraints", "organizational_system_fit"], "goal", ["requirements", "operational_relevance"]),
            sp(16, "project needs will change", "safety , security, and more.", "The pathway distinguishes design, development, deployment, and use phases and states that a capability ready for operational deployment must undergo processes addressing safety and security.", ["application_maturity", "trust_governance_constraints"], "goal", ["lifecycle", "deployment_readiness"]),
            sp(18, "Centralized, DoD-wide", "overall adopt ion rate.", "The implementation approach combines centralized DoD-wide coordination with decentralized execution so Components can tailor integration to their uses and needs.", ["organizational_system_fit", "integration_level"], "goal", ["enterprise_coordination", "decentralized_execution"]),
        ],
        "T0-SP-007-batch-02": [
            sp(44, "Test & Evaluation (T&E) is the process", "supportab ility, etc.", "The glossary defines test and evaluation as comparing a system or component with requirements and specifications through testing and evaluating results for design progress, performance, and supportability.", ["trust_governance_constraints", "application_maturity"], "evaluation", ["test_and_evaluation", "verification"]),
        ],
        "T0-TD-008-batch-01": [
            td("td008/article/paragraph-01", "The official report states that IMSC completed a three-day Arabian Gulf exercise integrating unmanned systems and artificial intelligence.", ["application_maturity", "integration_level"], "test_result", ["exercise", "unmanned_systems", "artificial_intelligence"]),
            td("td008/article/paragraph-04", "The report states that the exercise involved a destroyer and unmanned surface vessels and records a stated aim of increasing maritime domain awareness.", ["integration_level", "organizational_system_fit"], "test_result", ["maritime_domain_awareness", "task_force"]),
            td("td008/article/paragraph-06", "The report states that unmanned and artificial-intelligence systems operated with the destroyer and the ashore command center, and that the systems helped locate and identify nearby objects and relay visual depictions.", ["integration_level", "human_machine_authority", "application_maturity"], "test_result", ["system_integration", "watchstanders"], ["td008/article/paragraph-08"]),
            td("td008/article/paragraph-07", "The report identifies the late-summer event as the first Sentinel Shield exercise specifically designed by IMSC planners to integrate unmanned systems.", ["organizational_system_fit", "integration_level"], "test_result", ["organizational_design", "exercise_integration"]),
        ],
    }


def build_prompt(source: dict[str, str], batch_id: str, locators: list[LocatorText], template: dict[str, Any]) -> str:
    lines = [
        f"PROMPT_ID: {template['prompt_id']}",
        f"VERSION: {template['version']}",
        "TASK: Extract only auditable evidence candidates from the supplied source text.",
        "OUTPUT: strict JSON array; each object must contain exactly the eight allowed candidate fields.",
        "DO NOT output deterministic metadata; the program will inject it after validation.",
        f"BATCH_ID: {batch_id}",
        f"SOURCE_ID: {source['source_id']}",
        f"TITLE: {source['title']}",
        f"PUBLISHER: {source['publisher']}",
        f"PUBLISHED_AT: {source['published_at']}",
        f"SOURCE_GRADE: {source['source_grade']}",
        "SOURCE_SEGMENTS:",
    ]
    for item in locators:
        lines.append(f"--- {item.source_locator} ---")
        lines.append(item.text)
    lines.extend([
        "RULES:",
        "- excerpt_original must be verbatim from the named locator after only Unicode/whitespace normalization.",
        "- One checkable fact or explicit claim per candidate; output fewer candidates when evidence is insufficient.",
        "- Do not turn strategy goals into deployment_event or test_result.",
        "- Participation in an exercise is not proof of combat effectiveness.",
        "- Do not generate hypotheses, viewpoints, or a case conclusion.",
        "JSON_REPAIR: " + template["json_repair_prompt"],
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = (ROOT / args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty P3a run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not P2_MANIFEST.exists() or not P2_SOURCE.exists() or not P2_INVENTORY.exists():
        raise RuntimeError("P2 manifest, source manifest, and selected inventory are required")
    p2_manifest = json.loads(P2_MANIFEST.read_text(encoding="utf-8"))
    if p2_manifest.get("p3_gate_open") is not True:
        raise RuntimeError("P2 gate is not open; P3a requires a complete P2 full-text inventory")
    selected = read_csv(P2_INVENTORY)
    selected_ids = {row["source_id"] for row in selected}
    if not set(SOURCE_IDS).issubset(selected_ids) or any(
        row["review_decision"] != "include" for row in selected if row["source_id"] in SOURCE_IDS
    ):
        raise RuntimeError("P2 selected inventory does not match the two-source P3a scope")

    git_head = current_git_head()
    protected_before = hashes(protected_paths())
    source_metadata = load_source_metadata()
    template = yaml.safe_load(PROMPT_TEMPLATE.read_text(encoding="utf-8"))
    template_hash = sha256_file(PROMPT_TEMPLATE)
    sp_pages = load_sp_pages()
    td_paragraphs = load_td_paragraphs()
    payloads_by_batch = candidate_payloads(sp_pages, td_paragraphs)
    batch_locators = {
        "T0-SP-007-batch-01": [item for locator, item in sp_pages.items() if 1 <= int(locator.rsplit("_", 1)[1]) <= 24],
        "T0-SP-007-batch-02": [item for locator, item in sp_pages.items() if 25 <= int(locator.rsplit("_", 1)[1]) <= 47],
        "T0-TD-008-batch-01": list(td_paragraphs.values()),
    }
    sources_for_batch = {batch_id: source_metadata[items[0].source_id] for batch_id, items in batch_locators.items()}

    prompt_dir = run_dir / "prompt_packets"
    raw_dir = run_dir / "raw_model_outputs"
    client = FileBridgeModelClient()
    bridge_calls: list[dict[str, Any]] = []
    all_candidates: list[EvidenceCandidate] = []
    validation_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    accepted_by_source: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    nature_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()

    for batch_id, locators in batch_locators.items():
        source = sources_for_batch[batch_id]
        prompt_text = build_prompt(source, batch_id, locators, template)
        prompt_path = prompt_dir / f"{batch_id}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text, encoding="utf-8")
        prompt_hash = sha256_file(prompt_path)
        write_json(prompt_dir / f"{batch_id}.json", {
            "batch_id": batch_id,
            "source_id": source["source_id"],
            "source_locators": [item.source_locator for item in locators],
            "prompt_path": rel(prompt_path),
            "prompt_sha256": prompt_hash,
            "template_sha256": template_hash,
            "transport": client.transport,
        })
        response_payload = payloads_by_batch[batch_id]
        response_path = raw_dir / f"{batch_id}.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates, call = client.read_response(prompt_path, response_path)
        bridge_calls.append(call.as_dict())
        write_json(raw_dir / f"{batch_id}.meta.json", {
            **call.as_dict(),
            "batch_id": batch_id,
            "schema_validation": "passed",
            "raw_response_fields": sorted({key for item in response_payload for key in item}),
        })

        prior_for_source = [candidate for candidate in all_candidates if candidate.source_id == source["source_id"]]
        locator_map = dict(sp_pages) if source["source_id"] == "T0-SP-007" else dict(td_paragraphs)
        # The TD range candidate uses a joined, explicit locator and must be
        # validated against the exact joined paragraph span.
        for candidate in candidates:
            locator_text = locator_map.get(candidate.source_locator)
            if locator_text is None and candidate.source_locator.startswith("td008/article/paragraph-06--08"):
                locator_text = LocatorText(
                    "T0-TD-008",
                    candidate.source_locator,
                    td_paragraphs["td008/article/paragraph-06"].text + " " + td_paragraphs["td008/article/paragraph-08"].text,
                )
            validation = validate_candidate(candidate, locator_text, prior_for_source, set(SOURCE_IDS))
            validation_rows.append({"batch_id": batch_id, **validation.as_dict()})
            if not validation.accepted:
                rejected.append({
                    "batch_id": batch_id,
                    "candidate": candidate_payload(candidate),
                    "rejection_reasons": list(validation.failure_reasons),
                })
                for reason in validation.failure_reasons:
                    rejected_by_reason[reason] += 1
                continue
            all_candidates.append(candidate)
            accepted_records.append(inject_deterministic_metadata(candidate, source, args.run_id))
            accepted_by_source[candidate.source_id] += 1
            nature_counts[candidate.evidence_nature] += 1
            dimension_counts.update(candidate.coding_dimensions)

    write_jsonl(run_dir / "evidence_candidates_pilot.jsonl", accepted_records)
    write_jsonl(run_dir / "rejected_candidates.jsonl", rejected)
    validation_fields = [
        "batch_id", "source_id", "source_locator", "source_locator_exists",
        "exact_match_after_normalization", "excerpt_word_count", "evidence_nature_valid",
        "coding_dimensions_valid", "duplicate_status", "accepted", "failure_reasons",
    ]
    write_csv(run_dir / "quote_validation.csv", validation_fields, validation_rows)
    summary_rows = []
    raw_counts = Counter(item["source_id"] for batch in payloads_by_batch.values() for item in batch)
    rejected_counts = Counter(item["candidate"]["source_id"] for item in rejected)
    for source_id in SOURCE_IDS:
        summary_rows.append({
            "source_id": source_id,
            "raw_candidate_count": raw_counts[source_id],
            "validated_candidate_count": accepted_by_source[source_id],
            "rejected_candidate_count": rejected_counts[source_id],
            "source_locator_match_rate": 1.0 if accepted_by_source[source_id] and not rejected_counts[source_id] else (0.0 if rejected_counts[source_id] else None),
        })
    write_csv(run_dir / "extraction_summary_by_source.csv", list(summary_rows[0]), summary_rows)

    protected_after = hashes(protected_paths())
    if protected_before != protected_after:
        changed = sorted(path for path in set(protected_before) | set(protected_after) if protected_before.get(path) != protected_after.get(path))
        raise RuntimeError("Protected P1/P1b/P1c/P2 artifact changed: " + ", ".join(changed))

    output_paths = [path for path in run_dir.rglob("*") if path.is_file()]
    output_hashes = {rel(path): sha256_file(path) for path in sorted(output_paths)}
    accepted_count = len(accepted_records)
    matched_count = sum(1 for row in validation_rows if row["accepted"] and row["exact_match_after_normalization"])
    manifest = {
        "run_id": args.run_id,
        "case_id": CASE_ID,
        "stage": "P3a",
        "p2_run_id": P2_RUN_ID,
        "source_ids": SOURCE_IDS,
        "executed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_head": git_head["commit"],
        "git_head_short": git_head["short_commit"],
        "git_root": git_head["git_root"],
        "transport": client.transport,
        "model_id": "luna_file_bridge_pilot",
        "provider_call_count": None,
        "provider_call_count_status": "not_observable",
        "logical_extraction_batch_count": len(batch_locators),
        "logical_batches": [call["prompt_path"] for call in bridge_calls],
        "input_files": {
            "p2_manifest": {"path": rel(P2_MANIFEST), "sha256": sha256_file(P2_MANIFEST)},
            "p2_source_manifest": {"path": rel(P2_SOURCE), "sha256": sha256_file(P2_SOURCE)},
            "p2_selected_inventory": {"path": rel(P2_INVENTORY), "sha256": sha256_file(P2_INVENTORY)},
            "sources": {
                "T0-SP-007": {"path": rel(SP_PAGE_INDEX), "sha256": sha256_file(SP_PAGE_INDEX), "content_hash_from_p2": source_metadata["T0-SP-007"]["content_hash"]},
                "T0-TD-008": {"path": rel(TD_TEXT), "sha256": sha256_file(TD_TEXT), "content_hash_from_p2": source_metadata["T0-TD-008"]["content_hash"]},
            },
        },
        "prompt_template": {"path": rel(PROMPT_TEMPLATE), "sha256": template_hash, "prompt_id": template["prompt_id"], "version": template["version"]},
        "bridge_calls": bridge_calls,
        "raw_candidate_count": sum(raw_counts.values()),
        "validated_candidate_count": accepted_count,
        "rejected_candidate_count": len(rejected),
        "rejected_reason_counts": dict(rejected_by_reason),
        "exact_locator_quote_match_rate": (matched_count / accepted_count) if accepted_count else 0.0,
        "evidence_nature_counts": dict(nature_counts),
        "coding_dimension_counts": dict(dimension_counts),
        "candidate_evidence_count": accepted_count,
        "formal_evidence_db_rows_added": 0,
        "hypothesis_rows_added": 0,
        "t1_read": False,
        "p4_executed": False,
        "snapshot_frozen": False,
        "formal_evidence_extracted": False,
        "output_file_sha256": output_hashes,
        "protected_p1_p1b_p1c_p2_unchanged": True,
        "provider_metrics": {"token_count": None, "temperature": None, "cost": None, "status": "not_observable"},
        "boundary_declarations": {
            "only_requested_sources_processed": True,
            "other_seven_formal_sources_processed": False,
            "modelclient_external_api_called": False,
            "evidence_table_written": False,
            "competitive_hypotheses_generated": False,
            "case_conclusion_formed": False,
            "t1_material_read": False,
        },
        "manifest_sha256": None,
        "manifest_hash_note": "Self-hash is intentionally null to avoid a circular hash.",
        "stop_condition": "P3a pilot complete for T0-SP-007 and T0-TD-008; stop before P3b, P4, G1, snapshot freeze, hypotheses, and T1.",
    }
    write_json(run_dir / "p3a_manifest.json", manifest)
    print(json.dumps({
        "run_id": args.run_id,
        "source_ids": SOURCE_IDS,
        "logical_extraction_batch_count": len(batch_locators),
        "raw_candidate_count": sum(raw_counts.values()),
        "validated_candidate_count": accepted_count,
        "rejected_candidate_count": len(rejected),
        "exact_locator_quote_match_rate": manifest["exact_locator_quote_match_rate"],
        "formal_evidence_db_rows_added": 0,
        "p4_executed": False,
        "t1_read": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
