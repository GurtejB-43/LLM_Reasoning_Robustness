# LLM Reasoning Robustness

Authors: Gurtej Singh Bagga, Agastya Kompella, Rahul Menon

A research pipeline that uses the Toulmin Model of Argumentation to audit the structural coherence of LLM reasoning traces under adversarial input perturbations.

---

## Overview

The pipeline has three stages, each handled by a separate person:

**Stage 1 — Perturbation generation**
Samples 100 GSM8K and 150 StrategyQA examples, extracts implicit premises for StrategyQA, then generates three perturbed variants per example (premise deletion, contradiction injection, shuffling) using GPT-4o.

**Stage 2 — Model inference**
Runs `llama-3.3-70b-versatile` on all original and perturbed inputs via the Groq API. Outputs reasoning traces and final answers to `inference_results/`.

**Judge pipeline and analysis**
Applies a two-stage GPT-4o judge to extract Toulmin (Claim, Grounds, Warrant) triples from each reasoning trace and score warrant strength. Computes coherence scores, answer accuracy metrics, and generates all plots.

---

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

---

## Running the Judge Pipeline

Stages 1 and 2 have already been run. Their outputs are in `perturbed_questions/` and `inference_results/`. To run Stage 3:

```bash
python src/evaluation/judge.py

python src/evaluation/scoring.py

python src/evaluation/analysis.py
```

The judge script resumes automatically if interrupted — it skips any records already written to `results/scores/raw_judge_output.jsonl`.

To test on a small sample first:
```bash
python src/evaluation/judge.py --dataset gsm8k --limit 5
```

---

## Output

```
results/
  scores/
    raw_judge_output.jsonl   # per-trace Toulmin extractions and warrant scores
    scored_traces.jsonl      # one row per (example, condition) with all metrics
    per_example.jsonl        # one row per example with all conditions merged
  plots/
    cs_by_perturbation.png
    cs_drop_by_type.png
    cs_drop_vs_aad.png
    compromised_rate.png
```
