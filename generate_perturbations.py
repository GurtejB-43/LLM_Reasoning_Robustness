import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OpenAI_API_Key"))

# first define the JSON schema for all 3 perturbations
class PerturbedVariants(BaseModel):
    premise_deletion: str
    contradiction_injection: str
    shuffling: str

def generate_perturbations(input_file: str, output_file: str, dataset_type: str):
    print(f"Starting perturbations for {dataset_type}...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            data = json.loads(line)
            question = data.get("question", "")
            
            # now we formulate the prompt based on dataset type               
            if dataset_type == "StrategyQA":
                premises = data.get("implicit_premises", [])
                facts = data.get("original_facts", [])
                user_prompt = f"""
Original Question: {question}
Provided Facts: {facts}
Implicit Premises: {premises}

Your task is to generate three perturbed, standalone versions of this problem. For each version, write a self-contained paragraph that weaves the necessary facts/premises together and ends with the original question. 

1. premise_deletion: Write the full context paragraph, but explicitly REMOVE a crucial fact or premise necessary to reach the correct conclusion. 
Do NOT include any meta-commentary about the changes you made. Do NOT say things like "In this case/version," "Information has been removed," or "In this context/setting". The output must read as a natural, standalone paragraph that completely hides the fact that it has been tampered with.

2. contradiction_injection: Write the full context paragraph, but INJECT a statement that directly negates one of the provided facts or premises.
Do NOT include any meta-commentary about the changes you made. Do NOT say things like "In this case/version," "Information has been removed," or "In this context/setting". The output must read as a natural, standalone paragraph that completely hides the fact that it has been tampered with.

3. shuffling: Write the full context paragraph, but reorder the facts and premises to disrupt the logical flow, burying the relevant information among irrelevant noise.
Do NOT include any meta-commentary about the changes you made. Do NOT say things like "In this case/version," "Information has been removed," or "In this context/setting". The output must read as a natural, standalone paragraph that completely hides the fact that it has been tampered with.

CRITICAL RULE: Do NOT include any meta-commentary about the changes you made. 
Do NOT say things like "In this case/version," "Information has been removed," or "In this context/setting". Show, do not tell. 
The output must read as a natural, standalone paragraph that completely hides the fact that it has been tampered with.
"""
            else: # GSM8K
                user_prompt = f"""
Original Math Problem: {question}

Generate three perturbed versions of this math problem:
1. premise_deletion: Remove a crucial number or condition necessary to solve the problem.
2. contradiction_injection: Inject a mathematical condition that contradicts the existing numbers (e.g., stating a total that makes the individual parts impossible).
3. shuffling: Reorder the sentences and present the numbers out of chronological or logical order to confuse the reader.
"""

            system_prompt = "You are an expert dataset curator for an NLP robustness benchmark. Your task is to generate strict adversarial perturbations of reasoning questions. You must NOT include any meta-commentary about the changes you made. Do NOT say things like 'In this case/version,' 'Information has been removed,' or 'In this context/setting'. Show, do not tell. The output must read as a natural, standalone paragraph that completely hides the fact that it has been tampered with." 

            try:
                print(f"Perturbing {dataset_type} example {i+1}...")
                
                # Using gpt-5.4-mini as the model
                response = client.beta.chat.completions.parse(
                    model="gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=PerturbedVariants,
                    temperature=0.3 # Slight creativity allowed for generating natural-sounding perturbations
                )
                
                result = response.choices[0].message.parsed
                
                # combine original data with the 3 new conditions
                output_data = {
                    "original_id": i,
                    "dataset": dataset_type,
                    "conditions": {
                        "original": question,
                        "premise_deletion": result.premise_deletion,
                        "contradiction_injection": result.contradiction_injection,
                        "shuffled": result.shuffling
                    }
                }
                
                # For StrategyQA, keep the original metadata intact for reference
                if dataset_type == "StrategyQA":
                    output_data["metadata"] = {
                        "facts": facts,
                        "implicit_premises": premises
                    }
                
                outfile.write(json.dumps(output_data) + "\n")
                outfile.flush()
                
            except Exception as e:
                print(f"Error on {dataset_type} example {i+1}: {e}")
                continue

    print(f"Finished {dataset_type} perturbations!\n")

if __name__ == "__main__":
    # 1. Process StrategyQA
    sqa_input = "unperturbed_questions/strategyqa_premises_150.jsonl"
    sqa_output = "perturbed_questions/strategyqa_perturbed_150.jsonl"
    if os.path.exists(sqa_input):
        generate_perturbations(sqa_input, sqa_output, "StrategyQA")
    else:
        print(f"Could not find {sqa_input}. Please ensure the extraction script finished.")

    # 2. Process GSM8K
    gsm8k_input = "unperturbed_questions/gsm8k_sample_100.jsonl"
    gsm8k_output = "perturbed_questions/gsm8k_perturbed_100.jsonl"
    if os.path.exists(gsm8k_input):
        generate_perturbations(gsm8k_input, gsm8k_output, "GSM8K")
    else:
         print(f"Could not find {gsm8k_input}.")