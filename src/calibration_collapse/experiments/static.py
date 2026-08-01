"""Full-vignette static baseline.

Each case emits one ``evaluation.schema.TurnRecord`` with ``turn_bin='static'``:
the complete clinical record is handed over at once, and the model gives one
leading diagnosis plus one confidence score.
"""

from __future__ import annotations

from collections.abc import Callable

from calibration_collapse.checks.sameCases import load_manifest
from calibration_collapse.dataload.agentclinic import medQA
from calibration_collapse.evaluation.confidence import parse_probe
from calibration_collapse.evaluation.schema import TurnRecord, append_turn_record
from calibration_collapse.experiments.config import ExperimentConfig, load_config, select_cases
from calibration_collapse.models.run_model import query


def static_prompt(vignette: str) -> str:
    return (
        "You are a physician. The complete clinical record for a patient is below.\n"
        "Provide your single best leading diagnosis and a confidence score from 0 to 100.\n\n"
        f"{vignette}\n\n"
        'Respond only with JSON: {"diagnosis": "<leading diagnosis>", '
        '"confidence": <integer 0-100>}'
    )


def run_static(
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
            run_id = config.run_id(backbone, seed)
            for line_index, case in cases:
                entry = manifest.by_line_index(line_index)
                vignette = medQA.vignette(case["OSCE_Examination"])
                response = query_fn(backbone, static_prompt(vignette))
                diagnosis, confidence, ok, failure = parse_fn(response)
                append_turn_record(
                    config.output_path,
                    TurnRecord(
                        run_id=run_id,
                        case_id=entry.case_id,
                        condition="static",
                        backbone=backbone,
                        seed=seed,
                        turn_index=1,
                        total_turns=1,
                        turn_bin="static",
                        predicted_diagnosis=diagnosis,
                        confidence=confidence,
                        reference_diagnosis=entry.correct_diagnosis,
                        is_correct=None,
                        grading_method=None,
                        parse_success=ok,
                        failure_type=failure,
                        raw_probe_response=response,
                        evidence_seen=["complete vignette"],
                        evidence_order_seed=None,
                    ),
                )


def main(config_path: str = "configs/experiment.yaml") -> None:
    run_static(load_config(config_path))


if __name__ == "__main__":
    main()
