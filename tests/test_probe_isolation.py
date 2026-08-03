import copy

import pytest

from calibration_collapse.experiments.probe_isolation import (
    DialogueOutcome,
    PatientAgentConfig,
    assert_patient_agent_fixed,
    compare_probe_pairs,
    probe_without_history_mutation,
)
from calibration_collapse.experiments.config import load_config


BACKBONES = ("openai/gpt-4o-mini", "openai/o4-mini")
PATIENT = PatientAgentConfig(
    model="openai/gpt-4o-mini",
    temperature=0.05,
    max_output_tokens=200,
    prompt_version="agentclinic-patient-v1",
)


def make_outcome(case_id, backbone, probe_enabled, *, diagnosis="asthma", turns=6):
    events = tuple(
        {"role": "doctor" if index % 2 else "patient", "content": str(index)}
        for index in range(1, 2 * turns)
    )
    return DialogueOutcome(
        case_id=case_id,
        doctor_backbone=backbone,
        probe_enabled=probe_enabled,
        patient_config_fingerprint=PATIENT.fingerprint,
        turn_count=turns,
        final_diagnosis=diagnosis,
        dialogue_events=events,
        probe_responses=("Diagnosis: asthma\nConfidence: 80",) * turns
        if probe_enabled
        else (),
    )


def test_patient_agent_is_identical_for_both_backbones():
    selected = assert_patient_agent_fixed(
        {backbone: PATIENT for backbone in BACKBONES}
    )
    assert selected == PATIENT


def test_frozen_config_uses_one_patient_for_both_doctors():
    config = load_config("configs/experiment.yaml")
    assert config.backbones == BACKBONES
    assert config.patient_agent == PATIENT
    selected = assert_patient_agent_fixed(
        {backbone: config.patient_agent for backbone in config.backbones}
    )
    assert selected.fingerprint == PATIENT.fingerprint


def test_patient_agent_drift_is_rejected():
    changed = PatientAgentConfig(
        model="different/patient",
        temperature=PATIENT.temperature,
        max_output_tokens=PATIENT.max_output_tokens,
        prompt_version=PATIENT.prompt_version,
    )
    with pytest.raises(ValueError, match="configuration drift"):
        assert_patient_agent_fixed({BACKBONES[0]: PATIENT, BACKBONES[1]: changed})


def test_probe_receives_copy_and_cannot_mutate_live_history():
    history = [{"role": "doctor", "content": "Where is the pain?"}]
    original = copy.deepcopy(history)

    def destructive_probe(copied_history):
        copied_history.append({"role": "probe", "content": "private measurement"})
        return "Diagnosis: migraine\nConfidence: 40"

    response = probe_without_history_mutation(history, destructive_probe)
    assert response.startswith("Diagnosis:")
    assert history == original


def test_probe_output_in_dialogue_is_rejected():
    with pytest.raises(ValueError, match="must not appear"):
        DialogueOutcome(
            case_id="0",
            doctor_backbone=BACKBONES[0],
            probe_enabled=True,
            patient_config_fingerprint=PATIENT.fingerprint,
            turn_count=1,
            final_diagnosis="asthma",
            dialogue_events=({"role": "probe", "content": "hidden"},),
            probe_responses=("hidden",),
        )


def test_five_case_two_backbone_probe_ab_comparison_passes():
    controls = [
        make_outcome(str(case), backbone, False)
        for case in range(5)
        for backbone in BACKBONES
    ]
    probed = [
        make_outcome(str(case), backbone, True)
        for case in range(5)
        for backbone in BACKBONES
    ]
    report = compare_probe_pairs(controls, probed)

    assert report.case_count == 5
    assert report.backbone_count == 2
    assert report.pair_count == 10
    assert report.final_diagnosis_agreement == 10
    assert report.turn_count_agreement == 10
    assert report.exact_dialogue_agreement == 10
    assert report.passed


def test_probe_ab_comparison_reports_diagnosis_and_turn_changes():
    controls = [
        make_outcome(str(case), backbone, False)
        for case in range(5)
        for backbone in BACKBONES
    ]
    probed = [
        make_outcome(
            str(case),
            backbone,
            True,
            diagnosis="pneumonia" if case == 0 and backbone == BACKBONES[0] else "asthma",
            turns=5 if case == 1 and backbone == BACKBONES[1] else 6,
        )
        for case in range(5)
        for backbone in BACKBONES
    ]
    report = compare_probe_pairs(controls, probed)

    assert report.final_diagnosis_agreement == 9
    assert report.turn_count_agreement == 9
    assert not report.passed
