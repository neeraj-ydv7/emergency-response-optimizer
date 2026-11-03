#!/usr/bin/env bash
# Emergency Response Optimizer — one-command setup & run
set -e
cd "$(dirname "$0")/backend"

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q 2>/dev/null || pip install -r requirements.txt -q

echo "[2/3] Training the severity-triage model (skipped if already trained)..."
if [ ! -f "models/triage_model.joblib" ]; then
  python3 train_model.py
else
  echo "    models/triage_model.joblib already exists, skipping."
fi

echo "[3/3] Starting the server on http://localhost:8000 ..."
echo "    Login page:    http://localhost:8000/"
echo "    Accounts:      dispatcher / dispatch123   (or)   admin / admin123"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
