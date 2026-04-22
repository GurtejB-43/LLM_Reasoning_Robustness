'''
This file contains logic to sample data from the StrategyQA and GSMK8 datasets
and then 
'''
from datasets import load_dataset

# 1) StrategyQA
# HF repo broken, so load directly from the official GitHub raw URL 
url = "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json"
SQA_dataset = load_dataset("json", data_files=url, split="train")

# Shuffle with a fixed seed for reproducibility
SQA_shuffled = SQA_dataset.shuffle(seed=42)

# Select the first 150 examples
sampled_SQA_dataset = SQA_shuffled.select(range(150))

# Export to JSONL
sampled_SQA_dataset.to_json("strategyqa_sample_150.jsonl", orient="records", lines=True)
print("Successfully generated strategyqa_sample_150.jsonl")

# 2) GSM8K
# Load GSM8K with the required "main" configuration 
GSMK8_dataset = load_dataset("openai/gsm8k", "main", split="train")
GSMK8_shuffled = GSMK8_dataset.shuffle(seed=42)
sampled_GSMK8_dataset = GSMK8_shuffled.select(range(100))

sampled_GSMK8_dataset.to_json("gsm8k_sample_100.jsonl", orient="records", lines=True)
print("Successfully generated gsm8k_sample_100.jsonl")