import json

import pytest

from calibration_collapse.evaluation.schema import (
    SCHEMA_VERSION,
    TurnRecord,
    append_turn_record,
    read_turn_records,
    turn_bin_for,
)


def make_record(**overrides):
    values = {
        "run_id": "run-001",
        "case_id": "medqa-001",
        "condition": "sequential",
        "backbone": "provider/model",
        "seed": 0,
        "turn_index": 1,
        "total_turns": 4,
        "turn_bin": "early",
        "predicted_diagnosis": "asthma",
        "confidence": 70.0,
        "reference_diagnosis": "asthma",
        "is_correct": True,
        "grading_method": "normalized-exact-v1",
        "parse_success": True,
        "failure_type": None,
        "raw_probe_response": '{"diagnosis":"asthma","confidence":70}',
        "evidence_seen": ["wheezing"],
        "evidence_order_seed": None,
    }
    values.update(overrides)
    return TurnRecord(**values)


def test_four_turn_mapping_is_early_middle_late_late():
    assert [turn_bin_for(index, 4) for index in range(1, 5)] == [
        "early",
        "middle",
        "late",
        "late",
    ]


def test_frozen_six_turn_mapping_has_balanced_thirds():
    assert [turn_bin_for(index, 6) for index in range(1, 7)] == [
        "early",
        "early",
        "middle",
        "middle",
        "late",
        "late",
    ]


def test_each_pipeline_uses_the_same_record_type():
    sequential = make_record()
    randomized = make_record(
        condition="randomized_order",
        evidence_order_seed=23,
    )
    static = make_record(
        condition="static",
        turn_index=1,
        total_turns=1,
        turn_bin="static",
        evidence_seen=["complete vignette"],
    )

    assert {record.schema_version for record in (sequential, randomized, static)} == {
        SCHEMA_VERSION
    }
    expected_fields = set(sequential.to_dict())
    assert all(
        set(record.to_dict()) == expected_fields
        for record in (sequential, randomized, static)
    )


def test_static_turn_shape_is_enforced():
    with pytest.raises(ValueError, match="static records require"):
        make_record(condition="static", total_turns=4, turn_bin="early")


def test_randomized_condition_requires_order_seed():
    with pytest.raises(ValueError, match="require evidence_order_seed"):
        make_record(condition="randomized_order")


def test_failed_parse_retains_auditable_row():
    record = make_record(
        predicted_diagnosis="",
        confidence=None,
        is_correct=None,
        grading_method=None,
        parse_success=False,
        failure_type="invalid_json",
        raw_probe_response="not json",
    )
    assert record.failure_type == "invalid_json"


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "turns.jsonl"
    original = make_record()
    append_turn_record(path, original)

    assert read_turn_records(path) == [original]
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_reader_reports_bad_line_number(tmp_path):
    path = tmp_path / "turns.jsonl"
    path.write_text(make_record().to_json() + "\n{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_turn_records(path)
