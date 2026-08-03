"""Safeguards and paired analysis for the confidence-probe experiment.

The confidence probe is a measurement instrument, not a dialogue participant.
Its response is stored separately and must never be appended to the doctor,
patient, or measurement histories.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PatientAgentConfig:
    """Every setting that must stay fixed when the doctor backbone changes."""

    model: str
    temperature: float
    max_output_tokens: int
    prompt_version: str
    bias: str = "none"

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.prompt_version.strip():
            raise ValueError("patient model and prompt_version must not be empty")
        if self.max_output_tokens < 1:
            raise ValueError("patient max_output_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("patient temperature cannot be negative")

    @property
    def fingerprint(self) -> str:
        """Stable identity saved with every paired run."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_patient_agent_fixed(
    configs: Mapping[str, PatientAgentConfig],
) -> PatientAgentConfig:
    """Reject a study in which doctor backbones use different patients."""
    if not configs:
        raise ValueError("at least one doctor backbone is required")
    unique = {config.fingerprint for config in configs.values()}
    if len(unique) != 1:
        details = ", ".join(
            f"{backbone}={config.fingerprint}"
            for backbone, config in sorted(configs.items())
        )
        raise ValueError(f"patient-agent configuration drift: {details}")
    return next(iter(configs.values()))


def probe_without_history_mutation(
    dialogue_history: Any,
    probe_fn: Callable[[Any], Any],
) -> Any:
    """Run a probe on a deep copy and prove the live history did not change."""
    before = copy.deepcopy(dialogue_history)
    probe_input = copy.deepcopy(dialogue_history)
    response = probe_fn(probe_input)
    if dialogue_history != before:
        raise RuntimeError("confidence probe mutated live dialogue history")
    return response


@dataclass(frozen=True, slots=True)
class DialogueOutcome:
    """One completed case under either the probe or no-probe condition."""

    case_id: str
    doctor_backbone: str
    probe_enabled: bool
    patient_config_fingerprint: str
    turn_count: int
    final_diagnosis: str
    dialogue_events: tuple[dict[str, str], ...]
    probe_responses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not self.doctor_backbone:
            raise ValueError("case_id and doctor_backbone must not be empty")
        if self.turn_count < 1:
            raise ValueError("turn_count must be positive")
        if not self.final_diagnosis.strip():
            raise ValueError("final_diagnosis must not be empty")
        if any(event.get("role") == "probe" for event in self.dialogue_events):
            raise ValueError("probe output must not appear in dialogue_events")
        if self.probe_enabled and not self.probe_responses:
            raise ValueError("probe-enabled outcomes require separately logged probes")
        if not self.probe_enabled and self.probe_responses:
            raise ValueError("no-probe outcomes cannot contain probe responses")


@dataclass(frozen=True, slots=True)
class ProbePairComparison:
    case_id: str
    doctor_backbone: str
    same_final_diagnosis: bool
    same_turn_count: bool
    same_dialogue: bool
    without_probe_diagnosis: str
    with_probe_diagnosis: str
    without_probe_turn_count: int
    with_probe_turn_count: int


@dataclass(frozen=True, slots=True)
class ProbeIsolationReport:
    case_count: int
    backbone_count: int
    pair_count: int
    patient_config_fingerprint: str
    final_diagnosis_agreement: int
    turn_count_agreement: int
    exact_dialogue_agreement: int
    comparisons: tuple[ProbePairComparison, ...]

    @property
    def passed(self) -> bool:
        return (
            self.final_diagnosis_agreement == self.pair_count
            and self.turn_count_agreement == self.pair_count
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


def compare_probe_pairs(
    without_probe: Sequence[DialogueOutcome],
    with_probe: Sequence[DialogueOutcome],
    *,
    expected_case_count: int = 5,
) -> ProbeIsolationReport:
    """Compare matched A/B outcomes and enforce the five-case design."""
    if expected_case_count < 1:
        raise ValueError("expected_case_count must be positive")

    def index(
        outcomes: Sequence[DialogueOutcome], expected_probe: bool
    ) -> dict[tuple[str, str], DialogueOutcome]:
        indexed: dict[tuple[str, str], DialogueOutcome] = {}
        for outcome in outcomes:
            if outcome.probe_enabled is not expected_probe:
                raise ValueError("outcome placed in the wrong probe condition")
            key = (outcome.case_id, outcome.doctor_backbone)
            if key in indexed:
                raise ValueError(f"duplicate outcome for {key}")
            indexed[key] = outcome
        return indexed

    control = index(without_probe, False)
    probed = index(with_probe, True)
    if set(control) != set(probed):
        missing = sorted(set(control) - set(probed))
        extra = sorted(set(probed) - set(control))
        raise ValueError(f"unmatched probe pairs: missing={missing}, extra={extra}")

    cases = {case_id for case_id, _ in control}
    backbones = {backbone for _, backbone in control}
    if len(cases) != expected_case_count:
        raise ValueError(
            f"expected {expected_case_count} unique cases, found {len(cases)}"
        )
    expected_pairs = len(cases) * len(backbones)
    if len(control) != expected_pairs:
        raise ValueError("every case must be run with every doctor backbone")

    fingerprints = {
        outcome.patient_config_fingerprint
        for outcome in (*control.values(), *probed.values())
    }
    if len(fingerprints) != 1:
        raise ValueError("patient-agent configuration changed between paired runs")

    comparisons = tuple(
        ProbePairComparison(
            case_id=key[0],
            doctor_backbone=key[1],
            same_final_diagnosis=(
                control[key].final_diagnosis.strip().casefold()
                == probed[key].final_diagnosis.strip().casefold()
            ),
            same_turn_count=control[key].turn_count == probed[key].turn_count,
            same_dialogue=control[key].dialogue_events == probed[key].dialogue_events,
            without_probe_diagnosis=control[key].final_diagnosis,
            with_probe_diagnosis=probed[key].final_diagnosis,
            without_probe_turn_count=control[key].turn_count,
            with_probe_turn_count=probed[key].turn_count,
        )
        for key in sorted(control)
    )
    return ProbeIsolationReport(
        case_count=len(cases),
        backbone_count=len(backbones),
        pair_count=len(comparisons),
        patient_config_fingerprint=next(iter(fingerprints)),
        final_diagnosis_agreement=sum(
            comparison.same_final_diagnosis for comparison in comparisons
        ),
        turn_count_agreement=sum(
            comparison.same_turn_count for comparison in comparisons
        ),
        exact_dialogue_agreement=sum(
            comparison.same_dialogue for comparison in comparisons
        ),
        comparisons=comparisons,
    )


def write_probe_isolation_report(
    report: ProbeIsolationReport, path: str | Path
) -> None:
    """Write the auditable A/B comparison outside the tracked source tree."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
