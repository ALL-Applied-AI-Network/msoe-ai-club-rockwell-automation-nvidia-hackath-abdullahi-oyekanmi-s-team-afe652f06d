#!/usr/bin/env bash
# Mac / Linux: bash run_app.sh   (first run installs what the app needs into .venv)
cd "$(dirname "$0")"
PY=python3; command -v python3 >/dev/null || PY=python
if ! command -v "$PY" >/dev/null; then
  echo "Python is not installed. Install Python 3.10 or newer from https://www.python.org/downloads/"; exit 1
fi
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif ! "$PY" -c "import streamlit, cv2, pandas, requests, PIL, numpy" >/dev/null 2>&1; then
  echo "First run on this computer: installing what the app needs (1-3 minutes)..."
  "$PY" -m venv .venv && .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 \
    && .venv/bin/python -m pip install -r requirements.txt || { echo "Setup failed - check the internet connection and try again."; rm -rf .venv; exit 1; }
  PY=.venv/bin/python
fi
[ -f nvidia_key.txt ] && export NVIDIA_API_KEY="$(head -n1 nvidia_key.txt)"
[ -f ~/.streamlit/credentials.toml ] || { mkdir -p ~/.streamlit; printf '[general]\nemail = ""\n' > ~/.streamlit/credentials.toml; }
echo "Starting the website - open http://localhost:8501"
exec "$PY" -m streamlit run app.py
