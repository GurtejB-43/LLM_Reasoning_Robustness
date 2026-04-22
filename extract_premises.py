# Fault tolerant script that reads input file line by line 
# and appends the result to an output file

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OpenAI_API_Key"))

# Define the strict JSON schema we want GPT 5.4 mini to return
    # chose 5.4 mini instead of 4o because cheaper and likely better too
class StrategyQAPremises(BaseModel):
    original_question: str
    surfaced_premises: list[str]

# File paths
input_file = "unperturbed_questions/strategyqa_sample_150.jsonl"
output_file = "unperturbed_questions/strategyqa_premises_150.jsonl"

system_prompt = """You are an expert logician assisting with an NLP research project focused on Argumentation Theory. 
Your task is to analyze a multi-hop question, its ground truth answer, and the provided facts. 
Extract the strictly necessary *implicit premises* (Toulmin Warrants and unstated Grounds) required to logically deduce the answer. 
Do not include general trivia; only include the hidden logical steps that connect the concepts in the question."""

def extract_premises():
    # Open the output file in append mode ('a') so we don't overwrite progress if restarted
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'a', encoding='utf-8') as outfile:
        
        # Count lines to know where we are
        for i, line in enumerate(infile):
            data = json.loads(line)
            
            question = data.get("question", "")
            answer = data.get("answer", "")
            facts = data.get("facts", [])
            
            # Format the user prompt for this specific example
            user_prompt = f"""Analyze the following StrategyQA example:
Question: {question}
Ground Truth Answer: {answer}
Provided Facts: {facts}

Identify the implicit premises that a human or system must know to bridge the gap between the question and the answer."""

            try:
                print(f"Processing example {i+1}/150...")
                
                # Call the OpenAI API using the beta 'parse' endpoint for strict Structured Outputs
                response = client.beta.chat.completions.parse(
                    model="gpt-5.4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=StrategyQAPremises,
                    temperature=0.2 # Low temperature for highly analytical, deterministic outputs
                )
                
                # Extract the parsed object
                result = response.choices[0].message.parsed
                
                # Convert the Pydantic object to a dictionary, then to a JSON string
                output_json = result.model_dump()
                
                # Write the combined data to the new JSONL file
                output_data = {
                    "question": output_json["original_question"],
                    "answer": answer,
                    "original_facts": facts,
                    "implicit_premises": output_json["surfaced_premises"]
                }
                
                outfile.write(json.dumps(output_data) + "\n")
                outfile.flush() # Force write to disk immediately
                
            except Exception as e:
                print(f"Error on example {i+1}: {e}")
                # If an error occurs, we print it but keep the loop running
                continue

    print("Finished extracting premises!")

if __name__ == "__main__":
    extract_premises()