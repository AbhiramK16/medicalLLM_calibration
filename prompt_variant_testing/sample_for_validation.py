"""
Sample 20 records from variant_test_results.jsonl for hand-validation of the
parser.

Produces hand_validation_template.jsonl — which I then hand checked.
"""

import json
import random
from pathlib import Path

RESULTS_PATH = Path("variant_test_results.jsonl")
OUTPUT_PATH = Path("hand_validation_template.jsonl")
SAMPLE_SIZE = 20
SEED = 42  # fixed so the sample is reproducible


def main():
    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < SAMPLE_SIZE:
        raise SystemExit(
            f"Only {len(records)} records available, need at least {SAMPLE_SIZE}. "
            "Run run_variant_test.py first (or let it finish)."
        )

    random.seed(SEED)
    sample = random.sample(records, SAMPLE_SIZE)

    with open(OUTPUT_PATH, "w") as out_f:
        for rec in sample:
            template = {
                "case_id": rec["case_id"],
                "turn": rec["turn"],
                "variant": rec["variant"],
                "raw_response": rec["raw_response"],
                "parser_status": rec["parse_status"],
                "parser_diagnosis": rec["parsed_diagnosis"],
                "parser_confidence": rec["parsed_confidence"],
                # To fill in by hand
                "human_status": None,       # "parsed_ok" | "malformed" | "refused"
                "human_diagnosis": None,    # what I read as the stated diagnosis
                "human_confidence": None,   # what I read as the stated confidence
                "notes": "",                # anything odd worth flagging
            }
            out_f.write(json.dumps(template) + "\n")

    print(f"Wrote {SAMPLE_SIZE} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()