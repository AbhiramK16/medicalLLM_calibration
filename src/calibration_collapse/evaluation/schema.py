"""Versioned output contract shared by every experimental pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Mapping

SCHEMA_VERSION = 1

Condition = Literal["sequential", "randomized_order", "static"]
TurnBin = Literal["early", "middle", "late", "static"]

VALID_CONDITIONS = frozenset({"sequential", "randomized_order", "static"})
VALID_TURN_BINS = frozenset({"early", "middle", "late", "static"})


def turn_bin_for(turn_index: int, total_turns: int) -> TurnBin:
    """Return the preregistered proportional third for a dialogue turn."""
    if total_turns < 1:
        raise ValueError("total_turns must be at least 1")
    if not 1 <= turn_index <= total_turns:
        raise ValueError("turn_index must be between 1 and total_turns")

    fraction = turn_index / total_turns
    if fraction <= 1 / 3:
        return "early"
    if fraction <= 2 / 3:
        return "middle"
    return "late"


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One model prediction at one evidence/dialogue turn."""

    run_id: str
    case_id: str
    condition: Condition
    backbone: str
    seed: int
    turn_index: int
    total_turns: int
    turn_bin: TurnBin
    predicted_diagnosis: str
    confidence: float | None
    reference_diagnosis: str
    is_correct: bool | None
    grading_method: str | None
    parse_success: bool
    failure_type: str | None
    raw_probe_response: str
    evidence_seen: list[str]
    evidence_order_seed: int | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "case_id", "backbone", "reference_diagnosis"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.condition not in VALID_CONDITIONS:
            raise ValueError(f"unsupported condition: {self.condition}")
        if self.turn_bin not in VALID_TURN_BINS:
            raise ValueError(f"unsupported turn_bin: {self.turn_bin}")
        if self.total_turns < 1 or not 1 <= self.turn_index <= self.total_turns:
            raise ValueError("turn_index must be between 1 and total_turns")

        if self.condition == "static":
            if (self.turn_index, self.total_turns, self.turn_bin) != (1, 1, "static"):
                raise ValueError(
                    "static records require turn_index=1, total_turns=1, "
                    "and turn_bin='static'"
                )
            if self.evidence_order_seed is not None:
                raise ValueError("static records cannot have evidence_order_seed")
        else:
            expected_bin = turn_bin_for(self.turn_index, self.total_turns)
            if self.turn_bin != expected_bin:
                raise ValueError(
                    f"turn_bin must be {expected_bin!r} for turn "
                    f"{self.turn_index}/{self.total_turns}"
                )

        if self.condition == "randomized_order":
            if self.evidence_order_seed is None:
                raise ValueError(
                    "randomized_order records require evidence_order_seed"
                )
        elif self.evidence_order_seed is not None:
            raise ValueError(
                "evidence_order_seed is only valid for randomized_order records"
            )

        if self.confidence is not None and not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        if self.parse_success:
            if not self.predicted_diagnosis.strip() or self.confidence is None:
                raise ValueError(
                    "successful parses require a diagnosis and confidence"
                )
            if self.failure_type is not None:
                raise ValueError("successful parses cannot have failure_type")
        else:
            if self.confidence is not None:
                raise ValueError("failed parses require confidence=None")
            if not self.failure_type:
                raise ValueError("failed parses require failure_type")

        if self.is_correct is not None and not self.grading_method:
            raise ValueError("graded records require grading_method")
        if self.input_tokens < 0 or self.output_tokens < 0 or self.cost_usd < 0:
            raise ValueError("token counts and cost cannot be negative")
        if any(not isinstance(item, str) for item in self.evidence_seen):
            raise ValueError("evidence_seen must contain only strings")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize one compact, deterministically ordered JSONL row."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TurnRecord:
        """Validate and construct a record, rejecting unknown/missing fields."""
        expected = {field.name for field in fields(cls)}
        supplied = set(value)
        missing = expected - supplied
        unknown = supplied - expected
        if missing:
            raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
        return cls(**dict(value))


def append_turn_record(path: str | Path, record: TurnRecord) -> None:
    """Append exactly one validated record to a UTF-8 JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json())
        handle.write("\n")


def read_turn_records(path: str | Path) -> list[TurnRecord]:
    """Read and validate every nonblank JSONL row with line-aware errors."""
    records: list[TurnRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row must be a JSON object")
                records.append(TurnRecord.from_dict(value))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid turn record on line {line_number}: {error}") from error
    return records
