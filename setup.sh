#!/bin/bash
# Setup script for LLM Reasoning Robustness project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p results/scores results/plots

if [ ! -f .env ]; then
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "Created .env — add your OpenAI API key before running the judge pipeline."
fi

echo "Setup complete. Run: source venv/bin/activate"
