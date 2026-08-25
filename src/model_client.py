"""Minimal file-bridge model client used by the P3a pilot.

No network provider is contacted here.  A prompt packet is written by the
pipeline, the agent/Luna writes a JSON response file, and this bridge reads
and validates that response.  Provider-only metrics remain explicitly
unobservable rather than being inferred from logical batches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import EvidenceCandidate, model_validate


RAW_CANDIDATE_FIELDS = frozenset(
    {
        "source_id",
        "source_locator",
        "excerpt_original",
        "excerpt_zh",
        "normalized_claim",
        "coding_dimensions",
        "evidence_nature",
        "topics",
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class FileBridgeCall:
    transport: str
    prompt_path: str
    response_path: str
    prompt_sha256: str
    response_sha256: str
    candidate_count: int
    provider_call_count: None = None
    token_count: None = None
    temperature: None = None
    cost: None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "prompt_path": self.prompt_path,
            "response_path": self.response_path,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "candidate_count": self.candidate_count,
            "provider_call_count": self.provider_call_count,
            "token_count": self.token_count,
            "temperature": self.temperature,
            "cost": self.cost,
            "provider_metrics_status": "not_observable",
        }


class FileBridgeModelClient:
    """Read an agent-produced JSON file and enforce the P3 raw-output contract."""

    transport = "codex_agent_file_bridge"

    def read_response(self, prompt_path: Path, response_path: Path) -> tuple[list[EvidenceCandidate], FileBridgeCall]:
        prompt_bytes = prompt_path.read_bytes()
        response_bytes = response_path.read_bytes()
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"FileBridge response is not UTF-8 strict JSON: {response_path}") from exc
        if not isinstance(payload, list):
            raise ValueError("FileBridge response must be a JSON array")

        candidates: list[EvidenceCandidate] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"Candidate {index} is not a JSON object")
            actual = set(item)
            if actual != set(RAW_CANDIDATE_FIELDS):
                extra = sorted(actual - RAW_CANDIDATE_FIELDS)
                missing = sorted(RAW_CANDIDATE_FIELDS - actual)
                raise ValueError(f"Candidate {index} has extra={extra}, missing={missing}")
            candidates.append(model_validate(EvidenceCandidate, item))  # type: ignore[arg-type]

        call = FileBridgeCall(
            transport=self.transport,
            prompt_path=str(prompt_path),
            response_path=str(response_path),
            prompt_sha256=sha256_bytes(prompt_bytes),
            response_sha256=sha256_bytes(response_bytes),
            candidate_count=len(candidates),
        )
        return candidates, call

