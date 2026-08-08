"""
Analyze variant_test_results.jsonl:
  - print parser-extracted diagnosis, confidence, and response length for
    every (model, variant, case, turn)
  - compute per-(model, variant): avg response length, avg confidence,
    stdev of confidence, and an approximate accuracy

Everything is broken down by MODEL as well as variant, not just variant, to compare reasoning-tuned vs.
instruction-tuned backbone, or is specific to one of them. This matters
because the project's central comparison IS reasoning-tuned vs.
instruction-tuned -- a prompt variant that behaves inconsistently across
model types would inject a confound into that comparison.

ACCURACY CAVEAT:
There is no clean ground-truth diagnosis matcher yet; this is a rough keyword-overlap
heuristic ONLY, meant to give a directional read for picking a variant now.
It is NOT the project's real grader and should not be cited as such in any
writeup. Threshold (Jaccard >= 0.4, or subset containment)
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("variant_test_results_agentclinic.jsonl")
TEST_SET_PATH = Path("fixed_test_pairs_agentclinic.jsonl")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "with", "due", "to", "and", "or",
    "syndrome",  # generic enough across many diagnoses to skip for matching
}

# Jaccard similarity threshold for the heuristic matcher
MATCH_THRESHOLD = 0.4


def normalize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [w for w in text.split() if w and w not in STOPWORDS]
    return set(words)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def classify_match(ground_truth: str, predicted: str) -> tuple[str, float]:
    """Return (match_label, jaccard_score). match_label in {match, no_match}."""
    if not predicted:
        return "no_match", 0.0
    gt_set = normalize(ground_truth)
    pred_set = normalize(predicted)
    score = jaccard(gt_set, pred_set)

    substring_hit = bool(gt_set) and bool(pred_set) and (
        gt_set.issubset(pred_set) or pred_set.issubset(gt_set)
    )

    if score >= MATCH_THRESHOLD or substring_hit:
        return "match", score
    else:
        return "no_match", score


def check_format_compliance(variant: str, raw: str) -> bool:
    """
    Checks whether the raw response actually follows the structural
    constraint each prompt explicitly demands -- separate from whether the
    parser could FIND diagnosis/confidence anywhere in the text. Fully objective
    """
    stripped = raw.strip()
    if variant == "V1": # bunch of reasoning and then 2 lines of output
        lines = [l for l in stripped.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        return (
            lines[-2].lower().startswith("diagnosis:")
            and lines[-1].lower().startswith("confidence:")
        )
    elif variant == "V2": # 2 lines of output and then bunch of reasoning
        lines = [l for l in stripped.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        return (
            lines[0].lower().startswith("diagnosis:")
            and lines[1].lower().startswith("confidence:")
        )
    elif variant == "V3": #only 2 lines of output
        lines = [l for l in stripped.splitlines() if l.strip()]
        return (
            len(lines) == 2
            and lines[0].lower().startswith("diagnosis:")
            and lines[1].lower().startswith("confidence:")
        )
    return False


# MANUALLY COUNTED, not auto-detected. Keyed PER MODEL because this requires
# hand-reading actual diagnosis text, which differs by model
#
# Test applied during review: could this extra part appear as-is inside a
# standard clinical diagnosis label? -> NOT a violation, it's just naming.
# Or does it add something beyond a label -- a causal claim linking two
# separately-diagnosable conditions, or an unresolved alternative
# ("e.g., A or B")? -> a violation, this is reasoning/hedging.
#
# deepseek/deepseek-r1:
#   V1: Malignancy (recurrent testicular cancer or new primary malignancy)
#       Chronic lymphocytic leukemia/small lymphocytic lymphoma (CLL/SLL)
#   V2: Multiple System Atrophy (cerebellar variant, or MSA-C)
#       Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma (CLL/SLL)
#   V3: "Androgen-secreting tumor (e.g., adrenal or ovarian origin)"
#       Chronic lymphocytic leukemia/small lymphocytic lymphoma (CLL/SLL)
#
# openai/gpt-4o-mini:
#   V1: none
#   V2: Intestinal obstruction (possible volvulus or intussusception)
#       Acute Kidney Injury (AKI) likely due to a renal process
#       Malignancy (possible recurrence or new cancer)
#   V3: Intestinal obstruction (potentially volvulus or intussusception)
#       
MANUAL_EXTRANEOUS_DIAGNOSIS_CONTENT_COUNTS = {
    "deepseek/deepseek-r1": {"V1": 2, "V2": 2, "V3": 2},
    "openai/gpt-4o-mini": {"V1": 0, "V2": 3, "V3": 1},
}


def pearson_correlation(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2 or len(set(ys)) < 2 or len(set(xs)) < 2:
        return float("nan")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = (sum((x - mean_x) ** 2 for x in xs)) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    if std_x == 0 or std_y == 0:
        return float("nan")
    return cov / (std_x * std_y)


def main():
    ground_truth = {}
    with open(TEST_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ground_truth[row["case_id"]] = row.get("ground_truth_diagnosis")

    records = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    stats_lengths = defaultdict(list)
    stats_confidences = defaultdict(list)
    stats_turn_confidences = defaultdict(lambda: defaultdict(list))
    stats_match_counts = defaultdict(lambda: defaultdict(int))
    stats_format_compliant = defaultdict(list)
    stats_conf_by_match = defaultdict(lambda: defaultdict(list))
    stats_conf_match_pairs = defaultdict(lambda: ([], []))

    print(f"{'model':<20} {'variant':<6} {'case_id':<10} {'turn':<5} "
          f"{'confidence':<11} {'resp_len':<9} {'match':<9} {'diagnosis'}")
    print("-" * 120)

    for rec in records:
        model = rec["model"]
        variant = rec["variant"]
        case_id = rec["case_id"]
        turn = rec["turn"]
        raw = rec.get("raw_response") or ""
        diagnosis = rec.get("parsed_diagnosis")
        confidence = rec.get("parsed_confidence")
        resp_len = len(raw)
        key = (model, variant)

        gt = ground_truth.get(case_id)
        if gt is None or diagnosis is None:
            match_label, score = "no_match", 0.0
        else:
            match_label, score = classify_match(gt, diagnosis)

        stats_lengths[key].append(resp_len)
        if confidence is not None:
            stats_confidences[key].append(confidence)
            stats_turn_confidences[key][turn].append(confidence)
            stats_conf_by_match[key][match_label].append(confidence)
            xs, ys = stats_conf_match_pairs[key]
            xs.append(confidence)
            ys.append(1 if match_label == "match" else 0)
        stats_match_counts[key][match_label] += 1
        stats_format_compliant[key].append(check_format_compliance(variant, raw))

        print(f"{model:<20} {variant:<6} {case_id:<10} {turn:<5} "
              f"{str(confidence):<11} {resp_len:<9} {match_label:<9} {diagnosis}")

    all_keys = sorted(stats_lengths.keys())

    print("\n" + "=" * 120)
    print("AGGREGATE STATS PER MODEL x VARIANT")
    print("(accuracy = heuristic keyword-overlap match rate")
    print("=" * 120)
    print(f"{'model':<20} {'variant':<8} {'n':<5} {'avg_resp_len':<14} "
          f"{'avg_confidence':<16} {'stdev_confidence':<18} {'accuracy':<10}")

    for key in all_keys:
        model, variant = key
        lengths = stats_lengths[key]
        confidences = stats_confidences[key]
        counts = stats_match_counts[key]

        n = len(lengths)
        avg_len = statistics.mean(lengths) if lengths else float("nan")
        avg_conf = statistics.mean(confidences) if confidences else float("nan")
        stdev_conf = statistics.stdev(confidences) if len(confidences) > 1 else float("nan")

        matches = counts.get("match", 0)
        accuracy = matches / n if n else float("nan")

        print(f"{model:<20} {variant:<8} {n:<5} {avg_len:<14.1f} "
              f"{avg_conf:<16.3f} {stdev_conf:<18.3f} {accuracy:<10.1%}")

    print("\n" + "=" * 120)
    print("AVERAGE CONFIDENCE PER MODEL x VARIANT, PER TURN")
    print("=" * 120)

    all_turns = sorted({
        turn
        for turn_map in stats_turn_confidences.values()
        for turn in turn_map
    })
    header = f"{'model':<20} {'variant':<8}" + "".join(f"turn_{t:<10}" for t in all_turns)
    print(header)

    for key in all_keys:
        model, variant = key
        row = f"{model:<20} {variant:<8}"
        for t in all_turns:
            vals = stats_turn_confidences[key].get(t, [])
            avg = statistics.mean(vals) if vals else float("nan")
            row += f"{avg:<15.3f}"
        print(row)

    print("\n" + "=" * 120)
    print("ADDITIONAL STATS PER MODEL x VARIANT")
    print("=" * 120)
    print(f"{'model':<20} {'variant':<8} {'fmt_compl':<11} "
          f"{'extra_dx_content':<18} {'conf|match':<12} {'conf|no_match':<14} "
          f"{'gap':<8} {'conf_corr':<10} {'top_conf_share'}")

    for key in all_keys:
        model, variant = key
        n = len(stats_lengths[key])

        fmt_rate = (
            sum(stats_format_compliant[key]) / n if n else float("nan")
        )

        extra_count = (
            MANUAL_EXTRANEOUS_DIAGNOSIS_CONTENT_COUNTS.get(model, {}).get(variant, 0)
        )
        extra_rate = extra_count / n if n else float("nan")

        conf_match = stats_conf_by_match[key].get("match", [])
        conf_no_match = stats_conf_by_match[key].get("no_match", [])
        mean_conf_match = statistics.mean(conf_match) if conf_match else float("nan")
        mean_conf_no_match = (
            statistics.mean(conf_no_match) if conf_no_match else float("nan")
        )
        gap = (
            mean_conf_match - mean_conf_no_match
            if conf_match and conf_no_match
            else float("nan")
        )

        xs, ys = stats_conf_match_pairs[key]
        corr = pearson_correlation(xs, ys)

        confs = stats_confidences[key]
        if confs:
            most_common_count = max(confs.count(c) for c in set(confs))
            top_conf_share = most_common_count / len(confs)
        else:
            top_conf_share = float("nan")

        print(f"{model:<20} {variant:<8} {fmt_rate:<11.1%} {extra_rate:<18.1%} "
              f"{mean_conf_match:<12.3f} {mean_conf_no_match:<14.3f} "
              f"{gap:<8.3f} {corr:<10.3f} {top_conf_share:<.1%}")

    reviewed_models = {
        model for model, counts in MANUAL_EXTRANEOUS_DIAGNOSIS_CONTENT_COUNTS.items()
    }
    models_in_data = {model for model, _ in all_keys}
    unreviewed = models_in_data - reviewed_models
    if unreviewed:
        print(f"\nWARNING: no manual extra_dx_content review on record for: "
              f"{', '.join(sorted(unreviewed))}. Add an entry to "
              f"MANUAL_EXTRANEOUS_DIAGNOSIS_CONTENT_COUNTS.")

    print("\nNotes:")
    print("- fmt_compl: followed the EXACT structural instruction for that "
          "variant (not just parseable anywhere in the text).")
    print("- extra_dx_content: MANUALLY counted, PER MODEL, fraction of "
          "responses whose diagnosis field contains something beyond a "
          "standard clinical label.")
    print("- gap = conf|match - conf|no_match. Positive and larger is "
          "better -- means confidence tracks correctness. Near-zero or "
          "negative means the confidence score isn't meaningful.")
    print("- conf_corr: correlation between stated confidence and "
          "correctness across all records for that (model, variant) pair "
          "(NaN if no variance in the small sample)")
    print("- top_conf_share: fraction of responses sharing the single most "
          "common confidence value -- high share may indicate degenerate, "
          "non-discriminating confidence")


if __name__ == "__main__":
    main()