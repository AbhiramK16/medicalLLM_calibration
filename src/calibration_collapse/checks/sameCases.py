"""Mint and read the case manifest shared by every pipeline.

``case_id`` is the AgentClinic line index as a string, so it matches the native
``get_scenario(id)`` scheme. ``text_hash`` is a stable sha256 of the correct
diagnosis plus patient actor, recomputed at every pipeline stage to prove the
same case is being run in each condition. With line-index case IDs, ``verify_manifest``
is the only guard against upstream reordering silently changing what each id means.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from calibration_collapse.dataload.agentclinic import medQA

STATIC_FIELDS: tuple[str, ...] = (
    "Objective_for_Doctor",
    "Patient_Actor",
    "Physical_Examination_Findings",
    "Test_Results",
)

MANIFEST_VERSION = 1

@dataclass(frozen=True, slots=True)
class CaseEntry:
    case_id: str
    correct_diagnosis: str
    text_hash: str
    static_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.case_id.isdigit():
            raise ValueError(
                f"case_id must be a line index as string, got {self.case_id!r}"
            )
        if not self.correct_diagnosis.strip():
            raise ValueError("correct_diagnosis must not be empty")
        if len(self.text_hash) != 16:
            raise ValueError("text_hash must be a 16-char sha256 hex digest")
        if not self.static_fields:
            raise ValueError("static_fields must not be empty")


@dataclass(frozen=True, slots=True)
class CaseManifest:
    entries: tuple[CaseEntry, ...]

    def __post_init__(self) -> None:
        ids = [entry.case_id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case_id in manifest")

    def by_case_id(self, case_id: str) -> CaseEntry | None:
        for entry in self.entries:
            if entry.case_id == case_id:
                return entry
        return None

    def by_line_index(self, line_index: int) -> CaseEntry | None:
        return self.by_case_id(str(line_index))

    @property
    def case_ids(self) -> frozenset[str]:
        return frozenset(entry.case_id for entry in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def build_manifest() -> CaseManifest:
    loader = medQA()
    entries = [
        CaseEntry(
            case_id=str(i),
            correct_diagnosis=case["OSCE_Examination"]["Correct_Diagnosis"],
            text_hash=loader.findHash(case),
            static_fields=STATIC_FIELDS,
        )
        for i, case in enumerate(loader.data)
    ]
    return CaseManifest(tuple(entries))


def write_manifest(manifest: CaseManifest, path: str | Path) -> None:
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "entries": [asdict(entry) for entry in manifest.entries],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

def load_manifest(path: str | Path) -> CaseManifest:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest_version {payload.get('manifest_version')}"
        )
    entries = tuple(
        CaseEntry(
            **{**entry, "static_fields": tuple(entry["static_fields"])}
        )
        for entry in payload["entries"]
    )
    return CaseManifest(entries)


def verify_manifest(manifest: CaseManifest) -> None:
    """Fail loudly if the raw data file drifted from the manifest's snapshot."""
    fresh = build_manifest()
    if {entry.case_id for entry in fresh.entries} != {entry.case_id for entry in manifest.entries}:
        raise ValueError("case set changed since the manifest was built")
    for old, new in zip(manifest.entries, fresh.entries):
        if old.case_id != new.case_id:
            raise ValueError(f"case order changed at line {new.case_id}")
        if old.text_hash != new.text_hash:
            raise ValueError(
                f"content changed for case {old.case_id} "
                f"(was {old.text_hash}, now {new.text_hash})"
            )
