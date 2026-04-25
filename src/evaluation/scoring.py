import json
import re
from pathlib import Path

RAW_JUDGE_OUTPUT = Path("results/scores/raw_judge_output.jsonl")
SCORED_TRACES    = Path("results/scores/scored_traces.jsonl")
PER_EXAMPLE      = Path("results/scores/per_example.jsonl")

INFERENCE_FILES = {
    "gsm8k":       Path("inference_results/gsm8k_results.jsonl"),
    "strategyqa":  Path("inference_results/strategyqa_results.jsonl"),
}

GROUND_TRUTH_FILES = {
    "gsm8k":      Path("unperturbed_questions/gsm8k_sample_100.jsonl"),
    "strategyqa": Path("unperturbed_questions/strategyqa_sample_150.jsonl"),
}

PERTURBED_CONDITIONS = ["premise_deletion", "contradiction_injection", "shuffled"]
ALL_CONDITIONS       = ["original"] + PERTURBED_CONDITIONS


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(dataset: str) -> dict[int, str]:
    """
    Load ground truth answers keyed by line index (= original_id).

    GSM8K   : extract the integer after '####' in the answer field
    StrategyQA: map True/False boolean to 'yes'/'no'
    """
    gt = {}
    path = GROUND_TRUTH_FILES[dataset]
    with open(path) as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            if dataset == "gsm8k":
                answer_text = row.get("answer", "")
                match = re.search(r"####\s*([\d,]+)", answer_text)
                if match:
                    # Remove commas in numbers like 1,000
                    gt[idx] = match.group(1).replace(",", "").strip()
                else:
                    gt[idx] = None
            else:  # strategyqa
                raw = row.get("answer", None)
                if raw is True or (isinstance(raw, str) and raw.lower() == "true"):
                    gt[idx] = "yes"
                elif raw is False or (isinstance(raw, str) and raw.lower() == "false"):
                    gt[idx] = "no"
                else:
                    gt[idx] = None
    return gt


# ---------------------------------------------------------------------------
# is_correct derivation
# ---------------------------------------------------------------------------

def derive_is_correct(dataset: str, final_answer: str, ground_truth: str | None) -> bool:
    """
    Check whether the model's final_answer matches the ground truth answer.

    GSM8K      : extract the model's stated final number, then compare.
                 Priority: (1) $\boxed{N}$ pattern, (2) last integer in the text.
                 Avoids false positives from small numbers appearing in reasoning steps.
    StrategyQA : read the first word; nearly all outputs start with "Yes" or "No".
                 Fallback uses word-boundary regex to avoid matching "no" inside words.
    """
    if ground_truth is None or not final_answer:
        return False

    text = final_answer.strip()

    if dataset == "gsm8k":
        # Priority 1: explicit boxed answer e.g. $\boxed{16}$
        boxed = re.search(r"\$\\boxed\{([\d,]+)\}", text)
        if boxed:
            return boxed.group(1).replace(",", "") == ground_truth

        # Priority 2: "The final answer is: 16" or "= 16" at end of text
        final_stmt = re.search(
            r"(?:final answer is|the answer is)[^\d]*([\d,]+)\s*\.?\s*$",
            text, re.IGNORECASE
        )
        if final_stmt:
            return final_stmt.group(1).replace(",", "") == ground_truth

        # Priority 3: last integer in the entire text (most reliable fallback)
        all_ints = re.findall(r"\b(\d+)\b", text)
        if all_ints:
            return all_ints[-1].replace(",", "") == ground_truth

        return False

    else:  # strategyqa
        lower = text.lower().strip()
        first_word = lower.split()[0] if lower else ""
        if first_word in ("yes,", "yes.", "yes"):
            return ground_truth == "yes"
        if first_word in ("no,", "no.", "no"):
            return ground_truth == "no"

        has_yes = bool(re.search(r"\byes\b", lower))
        has_no  = bool(re.search(r"\bno\b",  lower))
        if has_yes and not has_no:
            return ground_truth == "yes"
        if has_no and not has_yes:
            return ground_truth == "no"
        return False


# ---------------------------------------------------------------------------
# CS computation
# ---------------------------------------------------------------------------

def compute_cs(scored_steps: list[dict]) -> float | None:
    """
    Coherence Score = mean warrant score across all steps.
    Returns None if there are no steps to score.
    """
    if not scored_steps:
        return None
    scores = [s["score"] for s in scored_steps]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Load inference results for final_answer and consistent_with_orig
# ---------------------------------------------------------------------------

def load_inference(dataset: str) -> dict[int, dict]:
    """
    Load inference results keyed by original_id.
    Returns dict: {original_id: {condition: {final_answer, consistent_with_orig}}}
    """
    data = {}
    with open(INFERENCE_FILES[dataset]) as f:
        for line in f:
            row = json.loads(line)
            oid = row["original_id"]
            data[oid] = {}
            for cond, res in row["results"].items():
                data[oid][cond] = {
                    "final_answer": res.get("final_answer", ""),
                    "consistent_with_orig": res.get("consistent_with_orig", None),
                }
    return data


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def run_scoring():
    """Compute all metrics and write scored_traces.jsonl and per_example.jsonl."""
    SCORED_TRACES.parent.mkdir(parents=True, exist_ok=True)

    judge_data: dict[str, dict[int, dict[str, dict]]] = {}
    with open(RAW_JUDGE_OUTPUT) as f:
        for line in f:
            row = json.loads(line)
            ds   = row["dataset"]
            oid  = row["original_id"]
            cond = row["condition"]
            judge_data.setdefault(ds, {}).setdefault(oid, {})[cond] = row

    datasets = list(judge_data.keys())

    all_scored_traces = []
    all_per_example   = []

    for dataset in datasets:
        print(f"\n=== Scoring {dataset.upper()} ===")
        gt_map    = load_ground_truth(dataset)
        inf_map   = load_inference(dataset)

        for original_id, conditions in sorted(judge_data[dataset].items()):
            gt = gt_map.get(original_id)

            condition_metrics: dict[str, dict] = {}
            for cond, judge_record in conditions.items():
                scored_steps = judge_record.get("scored_steps", [])
                cs = compute_cs(scored_steps)
                final_answer = inf_map.get(original_id, {}).get(cond, {}).get("final_answer", "")
                is_correct   = derive_is_correct(dataset, final_answer, gt)

                condition_metrics[cond] = {
                    "cs": cs,
                    "is_correct": is_correct,
                    "compromised": (cs < 0.5) if cs is not None else None,
                    "n_steps": len(scored_steps),
                    "final_answer": final_answer,
                    "ground_truth": gt,
                    "scored_steps": scored_steps,
                }

            orig = condition_metrics.get("original", {})
            cs_orig      = orig.get("cs")
            correct_orig = orig.get("is_correct", False)

            for cond in PERTURBED_CONDITIONS:
                if cond not in condition_metrics:
                    continue
                cm = condition_metrics[cond]

                cs_drop = (cs_orig - cm["cs"]) if (cs_orig is not None and cm["cs"] is not None) else None
                acc_drop = int(correct_orig) - int(cm["is_correct"])
                consistent = inf_map.get(original_id, {}).get(cond, {}).get("consistent_with_orig")

                cm["cs_drop"]              = cs_drop
                cm["accuracy_drop"]        = acc_drop
                cm["consistent_with_orig"] = consistent

                trace_row = {
                    "original_id":          original_id,
                    "dataset":              dataset,
                    "condition":            cond,
                    "cs":                   cm["cs"],
                    "cs_drop":              cs_drop,
                    "compromised":          cm["compromised"],
                    "is_correct":           cm["is_correct"],
                    "accuracy_drop":        acc_drop,
                    "consistent_with_orig": consistent,
                    "n_steps":              cm["n_steps"],
                    "ground_truth":         gt,
                }
                all_scored_traces.append(trace_row)

            if "original" in condition_metrics:
                cm = condition_metrics["original"]
                trace_row = {
                    "original_id": original_id,
                    "dataset":     dataset,
                    "condition":   "original",
                    "cs":          cm["cs"],
                    "cs_drop":     None,
                    "compromised": cm["compromised"],
                    "is_correct":  cm["is_correct"],
                    "accuracy_drop": None,
                    "consistent_with_orig": None,
                    "n_steps":     cm["n_steps"],
                    "ground_truth": gt,
                }
                all_scored_traces.append(trace_row)

            example_row = {
                "original_id": original_id,
                "dataset":     dataset,
                "ground_truth": gt,
                "conditions":  {}
            }
            for cond, cm in condition_metrics.items():
                example_row["conditions"][cond] = {
                    "cs":                   cm.get("cs"),
                    "cs_drop":              cm.get("cs_drop"),
                    "compromised":          cm.get("compromised"),
                    "is_correct":           cm.get("is_correct"),
                    "accuracy_drop":        cm.get("accuracy_drop"),
                    "consistent_with_orig": cm.get("consistent_with_orig"),
                    "n_steps":              cm.get("n_steps"),
                }
            all_per_example.append(example_row)

            orig_cs = condition_metrics.get("original", {}).get("cs")
            cs_str = f"{orig_cs:.3f}" if orig_cs is not None else "N/A"
            print(f"  id={original_id:3d} | orig_CS={cs_str} | gt={gt} | orig_correct={correct_orig}")

    with open(SCORED_TRACES, "w") as f:
        for row in all_scored_traces:
            f.write(json.dumps(row) + "\n")

    with open(PER_EXAMPLE, "w") as f:
        for row in all_per_example:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_scored_traces)} rows to {SCORED_TRACES}")
    print(f"Wrote {len(all_per_example)} rows to {PER_EXAMPLE}")


# ---------------------------------------------------------------------------
# Quick validation summary
# ---------------------------------------------------------------------------

def print_summary():
    """Print a quick sanity-check summary of the scored output."""
    import statistics

    rows = []
    with open(SCORED_TRACES) as f:
        for line in f:
            rows.append(json.loads(line))

    print("\n=== Scoring Summary ===")
    for dataset in ("gsm8k", "strategyqa"):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        orig_cs = [r["cs"] for r in ds_rows if r["condition"] == "original" and r["cs"] is not None]
        drops   = [r["cs_drop"] for r in ds_rows if r["cs_drop"] is not None]
        acc_drops = [r["accuracy_drop"] for r in ds_rows if r["accuracy_drop"] is not None]
        corrects = [r for r in ds_rows if r["condition"] == "original" and r["is_correct"]]

        print(f"\n  {dataset.upper()}")
        print(f"    Records:             {len(ds_rows)}")
        print(f"    Orig correct:        {len(corrects)}")
        if orig_cs:
            print(f"    Mean CS (orig):      {statistics.mean(orig_cs):.3f}")
        if drops:
            print(f"    Mean CS_drop:        {statistics.mean(drops):.3f}")
        if acc_drops:
            print(f"    Mean accuracy_drop:  {statistics.mean(acc_drops):.3f}")


if __name__ == "__main__":
    run_scoring()
    print_summary()
