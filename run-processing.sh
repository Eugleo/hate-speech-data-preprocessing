#!/bin/bash
pyenv local 3.11.2

echo "INFO: Python version should be 3.11.2, is $(python --version)"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python process_data.py