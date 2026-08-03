"""
Score the parser against your hand-checked labels in hand_validation_template.jsonl,
and log every disagreement explicitly to parser_failures.jsonl (never dropped
silently, per the task requirement).

ran AFTER filling in human_status / human_diagnosis / human_confidence for all 20 records.
"""

import json
from pathlib import Path

VALIDATION_PATH = Path("hand_validation_template.jsonl")
FAILURES_PATH = Path("parser_failures.jsonl")


def main():
    records = []
    with open(VALIDATION_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    unfilled = [r for r in records if r["human_status"] is None]
    if unfilled:
        raise SystemExit(
            f"{len(unfilled)} of {len(records)} records still have human_status "
            "= null. Fill in all 20 by hand before scoring."
        )

    n = len(records)
    status_matches = 0
    failures = []

    for rec in records:
        status_match = rec["human_status"] == rec["parser_status"]
        diagnosis_match = (
            rec["human_status"] != "parsed_ok"
            or (rec["human_diagnosis"] or "").strip().lower()
            == (rec["parser_diagnosis"] or "").strip().lower()
        )

        if status_match:
            status_matches += 1

        if not status_match or not diagnosis_match:
            failures.append({
                "case_id": rec["case_id"],
                "turn": rec["turn"],
                "variant": rec["variant"],
                "raw_response": rec["raw_response"],
                "parser_status": rec["parser_status"],
                "human_status": rec["human_status"],
                "parser_diagnosis": rec["parser_diagnosis"],
                "human_diagnosis": rec["human_diagnosis"],
                "parser_confidence": rec["parser_confidence"],
                "human_confidence": rec["human_confidence"],
                "notes": rec.get("notes", ""),
                "failure_type": (
                    "status_mismatch" if not status_match else "diagnosis_text_mismatch"
                ),
            })

    with open(FAILURES_PATH, "w") as out_f:
        for f_rec in failures:
            out_f.write(json.dumps(f_rec) + "\n")

    accuracy = status_matches / n
    print(f"Hand-checked: {n} records")
    print(f"Parser status accuracy: {status_matches}/{n} = {accuracy:.1%}")
    print(f"Logged {len(failures)} disagreement(s) to {FAILURES_PATH}")
    if failures:
        print("\nReview parser_failures.jsonl and decide whether the parser's "
              "regex/logic needs a fix before scaling to the full pipeline.")


if __name__ == "__main__":
    main()