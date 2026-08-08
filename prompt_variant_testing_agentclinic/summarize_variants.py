"""
Summarize parse_status counts per prompt variant from variant_test_results_agentclinic.jsonl.
Run after run_variant_test.py has completed (or partially completed).
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("variant_test_results_agentclinic.jsonl")


def main():
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            variant = rec["variant"]
            status = rec["parse_status"]
            counts[variant][status] += 1
            totals[variant] += 1

    print(f"{'Variant':<6} {'Total':<7} {'parsed_ok':<11} {'malformed':<11} "
          f"{'refused':<9} {'api_error':<10} {'ok_rate':<8}")
    print("-" * 70)
    for variant in sorted(counts.keys()):
        c = counts[variant]
        total = totals[variant]
        ok = c.get("parsed_ok", 0)
        mal = c.get("malformed", 0)
        ref = c.get("refused", 0)
        err = c.get("api_error", 0)
        ok_rate = f"{ok / total:.1%}" if total else "n/a"
        print(f"{variant:<6} {total:<7} {ok:<11} {mal:<11} {ref:<9} {err:<10} {ok_rate:<8}")


if __name__ == "__main__":
    main()