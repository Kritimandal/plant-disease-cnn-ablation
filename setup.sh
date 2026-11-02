#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m venv venv

if [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete. Start the dashboard with: streamlit run app/main.py"
