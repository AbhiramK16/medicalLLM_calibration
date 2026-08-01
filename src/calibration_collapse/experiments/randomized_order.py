"""Cumulative randomized-evidence baseline.

The same atomic evidence pieces that form the static vignette are revealed one
at a time in a reproducibly shuffled order (``evidence_order_seed``). Each
reveal emits one ``evaluation.schema.TurnRecord``.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from calibration_collapse.checks.sameCases import load_manifest
from calibration_collapse.dataload.agentclinic import medQA
from calibration_collapse.evaluation.confidence import parse_probe
from calibration_collapse.evaluation.schema import (
    TurnRecord,
    append_turn_record,
    turn_bin_for,
)
from calibration_collapse.experiments.config import ExperimentConfig, load_config, select_cases
from calibration_collapse.models.run_model import query


def randomized_prompt(evidence_seen: list[str]) -> str:
    return (
        "You are a physician. Clinical evidence is revealed to you one piece at a time.\n"
        "Evidence you have seen so far:\n"
        + "\n".join(f"- {item}" for item in evidence_seen)
        + "\n\nGive your current leading diagnosis and a confidence score from 0 to 100 "
        "based only on the evidence above.\n"
        'Respond only with JSON: {"diagnosis": "<leading diagnosis>", '
        '"confidence": <integer 0-100>}'
    )


def run_randomized(
    config: ExperimentConfig,
    *,
    query_fn: Callable = query,
    parse_fn: Callable = parse_probe,
) -> None:
    if not config.backbones:
        raise ValueError("config.backbones is empty; freeze it before a formal run")
    loader = medQA()
    manifest = load_manifest(config.manifest_path)
    cases = select_cases(loader, config)
    for backbone in config.backbones:
        for seed in config.seeds:
            rng = random.Random(seed)
            run_id = config.run_id(backbone, seed)
            for line_index, case in cases:
                entry = manifest.by_line_index(line_index)
                evidence = medQA.extract_evidence(case)
                if not evidence:
                    raise ValueError(f"case {entry.case_id} has no evidence pieces")
                shuffled = evidence.copy()
                rng.shuffle(shuffled)
                for turn_index in range(1, len(shuffled) + 1):
                    seen = shuffled[:turn_index]
                    response = query_fn(backbone, randomized_prompt(seen))
                    diagnosis, confidence, ok, failure = parse_fn(response)
                    append_turn_record(
                        config.output_path,
                        TurnRecord(
                            run_id=run_id,
                            case_id=entry.case_id,
                            condition="randomized_order",
                            backbone=backbone,
                            seed=seed,
                            turn_index=turn_index,
                            total_turns=len(shuffled),
                            turn_bin=turn_bin_for(turn_index, len(shuffled)),
                            predicted_diagnosis=diagnosis,
                            confidence=confidence,
                            reference_diagnosis=entry.correct_diagnosis,
                            is_correct=None,
                            grading_method=None,
                            parse_success=ok,
                            failure_type=failure,
                            raw_probe_response=response,
                            evidence_seen=seen,
                            evidence_order_seed=seed,
                        ),
                    )


def main(config_path: str = "configs/experiment.yaml") -> None:
    run_randomized(load_config(config_path))


if __name__ == "__main__":
    main()
