"""
Runs V1, V2, V3 elicitation prompts against the fixed (case, turn) test set
and logs raw + parsed output so refusal / malformed-output rate can be
compared across variants.

Usage:
    export OPENROUTER_API_KEY=sk-...
    python3 run_variant_test.py
"""

import json
import os
import re
import time
import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config — edit this block as the project evolves
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Placeholder models, anyone can choose what they want to test
MODELS = [
    "openai/gpt-4o-mini",
    # "deepseek/deepseek-r1",
]

TEST_SET_PATH = Path("fixed_test_pairs_v2.jsonl")
OUTPUT_PATH = Path("variant_test_results.jsonl")

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 60

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

VARIANTS = {
    "V1": (
        "You are a doctor whose job is to classify a medical diagnosis using "
        "case facts presented to you and your previous extensive medical "
        "expertise. Note that there might be little case information, but "
        "you still must provide the most likely diagnosis and reflect "
        "uncertainty through a lower confidence score rather than a refusal "
        "to answer. Based on the given information, write a single paragraph "
        "that explains your thought process and your reasoning behind your "
        "predicted diagnosis. Then, output two more lines following this "
        "format and immediately end your response:\n"
        "Diagnosis: <String (strictly one diagnosis, not multiple)>\n"
        "Confidence: <double from 0-1 (just a number, nothing else)>"
    ),
    "V2": (
        "A patient has come to you with details about a personal medical "
        "problem, and you have two tasks: determine the issue and state your "
        "confidence in your diagnosis, which will play a vital role in the "
        "patient's treatment. Use nothing but the facts given to you by the "
        "patient and your existing knowledge. Although you must choose one "
        "diagnosis, it is entirely possible that there is not enough "
        "information for you to make a confident guess, in which case you "
        "should adjust your confidence score.\n"
        "Address your response as if you are directly speaking to the "
        "patient. Start your response STRICTLY with this format:\n"
        "Diagnosis: <String (strictly one diagnosis, not multiple)>\n"
        "Confidence: <double from 0-1 (just a number, nothing else)>\n"
        "Then, explain to the patient why you chose your diagnosis and were "
        "as confident as you were, explaining uncertainties or other "
        "possible diagnoses."
    ),
    "V3": (
        "A professional medical clinic requires your expertise in "
        "deciphering a medical condition experienced by a patient. You are "
        "provided with information about the case, and your task is to come "
        "up with the right diagnosis for the patient along with your "
        "confidence score. Use your medical knowledge and the case facts, "
        "and think carefully about the given info before your response. "
        "Remember that there might be a lack of information which could "
        "make it difficult to decide; you must still choose one diagnosis "
        "but feel free to adjust your confidence accordingly.\n"
        "Your response should be in this exact format, NO OTHER TEXT\n"
        "Diagnosis: <String (strictly one diagnosis, not multiple)>\n"
        "Confidence: <double from 0-1 (just a number, nothing else)>"
    ),
}

# ---------------------------------------------------------------------------
# Parsing / classification
# ---------------------------------------------------------------------------

DIAGNOSIS_RE = re.compile(r"Diagnosis:\s*(.+)", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"Confidence:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# really important: the parser this project will use
def classify_response(raw_text: str) -> dict:
    """
    Return parse status + extracted fields for one raw model response.
 
    Status is one of:
      api_error                          - the API call itself failed
      malformed_missing_both_fields      - neither Diagnosis: nor Confidence: found
      malformed_missing_diagnosis_field  - Confidence: found, Diagnosis: missing
      malformed_missing_confidence_field - Diagnosis: found, Confidence: missing
      malformed_confidence_not_numeric   - Confidence: value isn't parseable as a float
      malformed_confidence_out_of_range  - Confidence: value parsed but outside [0, 1]
      malformed_empty_diagnosis          - Diagnosis: field present but blank
      parsed_ok                          - everything extracted cleanly
 
    Note: a missing diagnosis field (malformed_missing_diagnosis_field /
    malformed_missing_both_fields) covers refusals too -- a model that
    declines to answer won't have a Diagnosis: field, so it lands in one
    of those buckets rather than a separate "refused" status.
    """
    if raw_text is None:
        return {"status": "api_error", "diagnosis": None, "confidence": None}
 
    diag_match = DIAGNOSIS_RE.search(raw_text)
    conf_match = CONFIDENCE_RE.search(raw_text)
 
    if not diag_match and not conf_match:
        return {"status": "malformed_missing_both_fields", "diagnosis": None, "confidence": None}
    if not diag_match:
        return {"status": "malformed_missing_diagnosis_field", "diagnosis": None, "confidence": None}
    if not conf_match:
        return {"status": "malformed_missing_confidence_field", "diagnosis": None, "confidence": None}
 
    diagnosis = diag_match.group(1).strip().split("\n")[0].strip()
    conf_str = conf_match.group(1).strip()
 
    try:
        confidence = float(conf_str)
    except ValueError: #confidence not a number
        return {"status": "malformed_confidence_not_numeric", "diagnosis": diagnosis, "confidence": None}
 
    if not (0.0 <= confidence <= 1.0): # confidence out of range
        return {"status": "malformed_confidence_out_of_range", "diagnosis": diagnosis, "confidence": confidence}
 
    if not diagnosis: # would go here if the model didnt put anything after Diagnosis: 
        return {"status": "malformed_empty_diagnosis", "diagnosis": None, "confidence": confidence}
 
    return {"status": "parsed_ok", "diagnosis": diagnosis, "confidence": confidence}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_openrouter(model: str, system_prompt: str, case_text: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case_text},
        ],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 — log every failure explicitly
            print(f"  [attempt {attempt}/{MAX_RETRIES}] error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    if not OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY before running.")

    rows = []
    with open(TEST_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} (case, turn) rows.")
    print(f"Running {len(MODELS)} model(s) x {len(VARIANTS)} variant(s) "
          f"= {len(MODELS) * len(VARIANTS) * len(rows)} total calls.\n")

    with open(OUTPUT_PATH, "a") as out_f:
        for model in MODELS: #loop per model
            for variant_name, system_prompt in VARIANTS.items(): #loop per prompt variant
                for row in rows:
                    print(f"[{model} | {variant_name}] case={row['case_id']} "
                          f"turn={row['turn']}")
                    raw = call_openrouter(model, system_prompt, row["cumulative_text"])
                    parsed = classify_response(raw)

                    record = {
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "model": model,
                        "variant": variant_name,
                        "case_id": row["case_id"],
                        "turn": row["turn"],
                        "total_turns": row["total_turns"],
                        "raw_response": raw,
                        "parse_status": parsed["status"],
                        "parsed_diagnosis": parsed["diagnosis"],
                        "parsed_confidence": parsed["confidence"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

    print(f"\nDone. Results appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()