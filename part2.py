import os
import json
import re
import time
from groq import Groq

# Configure API keys in order of usage.
# Preferred: set GROQ_API_KEYS="key1,key2,key3" in your environment.
API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    single_key = os.getenv("GROQ_API_KEY", "")
    if single_key:
        API_KEYS = [single_key]
if not API_KEYS:
    raise ValueError("No Groq API keys found. Set GROQ_API_KEYS or GROQ_API_KEY.")

# Model IDs to be used for the project
MODELS = {
    "deepseek": "llama-3.3-70b-versatile",
    "llama": "llama-3.1-8b-instant"
}


class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.index = 0
        self.client = Groq(api_key=self.api_keys[self.index])

    def current_label(self):
        return f"key#{self.index + 1}"

    def switch_to_next_key(self):
        if self.index + 1 >= len(self.api_keys):
            return False
        self.index += 1
        self.client = Groq(api_key=self.api_keys[self.index])
        return True


key_manager = APIKeyManager(API_KEYS)


def _parse_retry_after_seconds(error_text):
    """
    Parses retry delay from strings like:
    "Please try again in 7m32.736s"
    "Please try again in 45s"
    Returns float seconds or None when parsing fails.
    """
    match = re.search(r"Please try again in\s+((?:(\d+)m)?([\d.]+)s)", error_text)
    if not match:
        return None

    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0.0
    return minutes * 60 + seconds


def _is_rate_limit_error(error_text):
    lowered = error_text.lower()
    return "429" in lowered or "rate limit" in lowered or "rate_limit_exceeded" in lowered


def load_processed_ids(output_path):
    processed_ids = set()
    if not os.path.exists(output_path):
        return processed_ids

    with open(output_path, 'r') as outfile:
        for line in outfile:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_id = row.get("original_id")
            if row_id is not None:
                processed_ids.add(str(row_id))

    return processed_ids


def process_dataset(input_path, output_path, model_key):
    model_id = MODELS[model_key]
    is_ds = (model_key == "deepseek")
    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        print(f"Resuming run. Found {len(processed_ids)} already-processed IDs in {output_path}.")

    with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
        for line in infile:
            data = json.loads(line)
            row_id = str(data.get("original_id"))
            if row_id in processed_ids:
                continue

            conditions = data.get("conditions", {})
            ground_truth = data.get("answer", "N/A") # Ensure ground truth is preserved

            output_entry = {
                "original_id": data.get("original_id"),
                "model": model_id,
                "results": {}
            }

            for cond_name, prompt in conditions.items():
                print(f"Processing {model_key} - {cond_name} for ID {data.get('original_id')}...")

                trace, final_ans = get_inference(model_id, prompt, is_deepseek=is_ds)

                output_entry["results"][cond_name] = {
                    "reasoning_trace": trace,
                    "final_answer": final_ans,
                    "is_correct": (str(ground_truth).lower() in str(final_ans).lower())
                }

                # Rate limiting to respect Groq's developer tier
                time.sleep(2)

            # Add Consistency and Accuracy Drop for this specific example
            orig_correct = output_entry["results"]["original"]["is_correct"]
            orig_ans = output_entry["results"]["original"]["final_answer"]

            for cond in ["premise_deletion", "contradiction_injection", "shuffled"]:
                res = output_entry["results"][cond]
                res["consistent_with_orig"] = (res["final_answer"] == orig_ans)
                # AAD for this row: 1 if original was right and this is wrong, -1 if vice versa, 0 if same
                res["accuracy_drop"] = int(orig_correct) - int(res["is_correct"])

            outfile.write(json.dumps(output_entry) + "\n")
            outfile.flush()
            processed_ids.add(row_id)


def get_inference(model_id, prompt, is_deepseek=False):
    """
    Fetches model response and extracts reasoning trace and final answer.
    """
    system_msg = "Think step by step before providing your final answer." if not is_deepseek else ""

    # Keep retrying on rate-limit responses until the request is accepted.
    while True:
        try:
            response = key_manager.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # Deterministic for research consistency
            )
            full_content = response.choices[0].message.content

            if is_deepseek:
                # Extract content between <think> tags for DeepSeek-R1
                think_match = re.search(r'<think>(.*?)</think>', full_content, re.DOTALL)
                trace = think_match.group(1).strip() if think_match else ""
                answer = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
            else:
                # LLaMA CoT: The whole response is the trace;
                # common practice is to take the last sentence as the answer
                trace = full_content
                answer = full_content.split('.')[-2] if '.' in full_content else full_content

            return trace, answer
        except Exception as e:
            err_text = str(e)
            if _is_rate_limit_error(err_text):
                prev_key_label = key_manager.current_label()
                switched = key_manager.switch_to_next_key()
                if switched:
                    print(
                        f"Rate limit hit for {model_id} on {prev_key_label}. "
                        f"Switching to {key_manager.current_label()} and retrying..."
                    )
                    continue

                wait_seconds = _parse_retry_after_seconds(err_text)
                # Fallback wait when server does not provide a parseable delay.
                if wait_seconds is None:
                    wait_seconds = 30.0

                # Add a small safety buffer to avoid retrying too early.
                wait_seconds += 1.0
                print(
                    f"Rate limit hit for {model_id} on last available {key_manager.current_label()}. "
                    f"Sleeping {wait_seconds:.1f}s before retrying..."
                )
                time.sleep(wait_seconds)
                continue

            print(f"Error during inference: {e}")
            return None, None

def evaluate_metrics(results):
    """
    Calculates Answer Accuracy Drop (AAD) and Answer Consistency Rate (ACR).
    AAD = Accuracy(Original) - Accuracy(Perturbed)
    ACR = % of perturbed answers that match the model's own original answer.
    """
    # This logic assumes 'results' is a list of objects for one specific model/dataset
    # comparing 'original' vs 'perturbed' conditions.
    pass 

if __name__ == "__main__":
    strategyqa_output = "inference_results/strategyqa_results.jsonl"
    gsm8k_output = "inference_results/gsm8k_results.jsonl"

    if not os.path.exists(strategyqa_output) or os.path.getsize(strategyqa_output) == 0:
        # StrategyQA using the perturbed file generated in Stage 1
        process_dataset(
            "perturbed_questions/strategyqa_perturbed_150.jsonl",
            strategyqa_output,
            "deepseek"
        )
    else:
        print(f"Skipping StrategyQA because {strategyqa_output} already exists.")

    # GSM8K using the perturbed file generated in Stage 1
    process_dataset(
        "perturbed_questions/gsm8k_perturbed_100.jsonl",
        gsm8k_output,
        "deepseek"
    )