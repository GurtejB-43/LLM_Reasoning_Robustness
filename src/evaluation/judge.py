import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = "gpt-4o"

INFERENCE_FILES = {
    "gsm8k": Path("inference_results/gsm8k_results.jsonl"),
    "strategyqa": Path("inference_results/strategyqa_results.jsonl"),
}

OUTPUT_FILE = Path("results/scores/raw_judge_output.jsonl")

CONDITIONS = ["original", "premise_deletion", "contradiction_injection", "shuffled"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a formal logic annotator working on an NLP research project.
Your task is to decompose an LLM's step-by-step reasoning trace into a structured
Toulmin argumentation graph.

For each distinct reasoning step in the trace, extract exactly one triple:
  - claim:   the specific conclusion or sub-conclusion being asserted in this step
  - grounds: the evidence or data points (from the prompt or prior steps) that support the claim
  - warrant: the logical bridge — the implicit rule or principle that explains WHY the grounds justify the claim

Rules:
- Only extract steps that contain a substantive logical move. Skip filler phrases.
- Each triple must be self-contained and reflect one coherent reasoning step.
- Return a JSON object with a single key "steps" whose value is a list of
  {{"claim": "...", "grounds": "...", "warrant": "..."}} objects.
- If the trace has no extractable reasoning steps, return {{"steps": []}}.
- Do not add any commentary outside the JSON."""

EXTRACTION_USER = """Reasoning trace to decompose:

{trace}

Return the Toulmin triples as JSON."""

SCORING_SYSTEM = """You are a formal logic evaluator working on an NLP research project.
You will be given a single Toulmin reasoning triple (Claim, Grounds, Warrant) extracted
from an LLM's reasoning trace.

Score the logical strength of the Warrant on a scale from 0.0 to 1.0:
  1.0 — the warrant is logically valid and fully justifies the claim given the grounds
  0.75 — the warrant is mostly sound with minor gaps
  0.5  — the warrant is partially valid but relies on unstated assumptions or weak logic
  0.25 — the warrant is largely flawed, speculative, or only superficially relevant
  0.0  — the warrant is entirely invalid, fabricated, or the claim does not follow from the grounds

Additionally flag whether this warrant appears to incorporate a distractor or injected
contradiction from the input (distractor_incorporated: true/false).

Return a JSON object with exactly these keys:
  {{"score": <float 0.0-1.0>, "justification": "<one sentence>", "distractor_incorporated": <bool>}}"""

SCORING_USER = """Toulmin triple to evaluate:

Claim:   {claim}
Grounds: {grounds}
Warrant: {warrant}

Return the evaluation as JSON."""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _call_with_retry(messages: list[dict], max_retries: int = 5) -> str:
    """
    Call GPT-4o with exponential backoff on rate-limit errors.
    Returns the raw response text.
    """
    delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            is_last = attempt == max_retries - 1
            if is_last:
                raise
            if is_rate_limit:
                print(f"      Rate limit hit — sleeping {delay:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
            else:
                print(f"      API error: {err[:120]} — retrying in {delay:.0f}s")
                time.sleep(delay)
    raise RuntimeError("Exhausted retries")


# ---------------------------------------------------------------------------
# Stage 3a — Extraction
# ---------------------------------------------------------------------------

def extract_steps(trace: str) -> list[dict]:
    """
    Decompose a reasoning trace into Toulmin (claim, grounds, warrant) triples.
    Returns a list of dicts; empty list if trace has no extractable steps.
    """
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_USER.format(trace=trace)},
    ]
    raw = _call_with_retry(messages)
    try:
        parsed = json.loads(raw)
        steps = parsed.get("steps", [])
        # Validate each step has the required keys
        valid = []
        for s in steps:
            if all(k in s for k in ("claim", "grounds", "warrant")):
                valid.append({
                    "claim": str(s["claim"]),
                    "grounds": str(s["grounds"]),
                    "warrant": str(s["warrant"]),
                })
        return valid
    except json.JSONDecodeError:
        print(f"      Warning: could not parse extraction JSON — returning empty steps")
        return []


# ---------------------------------------------------------------------------
# Stage 3b — Scoring
# ---------------------------------------------------------------------------

def score_step(step: dict) -> dict:
    """
    Score the logical strength of a single Toulmin warrant.
    Returns dict with keys: score (float), justification (str), distractor_incorporated (bool).
    """
    messages = [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": SCORING_USER.format(
            claim=step["claim"],
            grounds=step["grounds"],
            warrant=step["warrant"],
        )},
    ]
    raw = _call_with_retry(messages)
    try:
        parsed = json.loads(raw)
        return {
            "score": float(parsed.get("score", 0.0)),
            "justification": str(parsed.get("justification", "")),
            "distractor_incorporated": bool(parsed.get("distractor_incorporated", False)),
            "claim": step["claim"],
            "grounds": step["grounds"],
            "warrant": step["warrant"],
        }
    except (json.JSONDecodeError, ValueError):
        print(f"      Warning: could not parse scoring JSON — assigning score 0.0")
        return {
            "score": 0.0,
            "justification": "parse error",
            "distractor_incorporated": False,
            "claim": step["claim"],
            "grounds": step["grounds"],
            "warrant": step["warrant"],
        }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_processed_keys(output_path: Path) -> set[str]:
    """Return set of 'original_id|dataset|condition' strings already in the output file."""
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = f"{row['original_id']}|{row['dataset']}|{row['condition']}"
                processed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_judge(datasets: list[str], limit: int | None = None):
    """
    Run the two-stage judge on all records in the specified datasets.
    Writes one output record per (original_id, dataset, condition) to OUTPUT_FILE.
    """
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed_keys(OUTPUT_FILE)
    if processed:
        print(f"Resuming — {len(processed)} records already in output file.")

    with open(OUTPUT_FILE, "a") as out_f:
        for dataset in datasets:
            input_path = INFERENCE_FILES[dataset]
            print(f"\n=== Processing {dataset.upper()} ({input_path}) ===")

            with open(input_path) as in_f:
                lines = in_f.readlines()

            if limit is not None:
                lines = lines[:limit]

            for line_idx, line in enumerate(lines):
                record = json.loads(line)
                original_id = record["original_id"]
                model = record["model"]

                for condition in CONDITIONS:
                    key = f"{original_id}|{dataset}|{condition}"
                    if key in processed:
                        continue

                    result_entry = record["results"].get(condition)
                    if result_entry is None:
                        print(f"  [{dataset}] id={original_id} cond={condition} — missing, skipping")
                        continue

                    # Bug fix: reasoning is in final_answer, not reasoning_trace
                    trace_text = result_entry.get("final_answer", "").strip()
                    if not trace_text:
                        print(f"  [{dataset}] id={original_id} cond={condition} — empty trace, skipping")
                        continue

                    print(f"  [{dataset}] id={original_id:3d} ({line_idx+1}/{len(lines)}) cond={condition}")

                    # Stage 3a — extract Toulmin triples
                    steps = extract_steps(trace_text)
                    print(f"      3a: extracted {len(steps)} steps")

                    # Stage 3b — score each step
                    scored_steps = []
                    for step_idx, step in enumerate(steps):
                        scored = score_step(step)
                        scored_steps.append(scored)
                        print(f"      3b: step {step_idx+1}/{len(steps)} score={scored['score']:.2f}")
                        time.sleep(0.5)  # gentle pacing between step calls

                    output_record = {
                        "original_id": original_id,
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "trace_text": trace_text,
                        "scored_steps": scored_steps,
                    }

                    out_f.write(json.dumps(output_record) + "\n")
                    out_f.flush()
                    processed.add(key)

                    # Brief pause between conditions to respect rate limits
                    time.sleep(1.0)

    print(f"\nDone. Output written to {OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage GPT-4o Toulmin judge pipeline")
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "strategyqa", "both"],
        default="both",
        help="Which dataset to process (default: both)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records per dataset (for testing)",
    )
    args = parser.parse_args()

    datasets = ["gsm8k", "strategyqa"] if args.dataset == "both" else [args.dataset]
    run_judge(datasets=datasets, limit=args.limit)
