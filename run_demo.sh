#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo " Starting AI Skin: Intelligent Skin Diseases Detection"
echo "=========================================================="

# Activate Python venv
source venv/bin/activate

# Start backend in background
echo "[*] Starting FastAPI Backend on http://127.0.0.1:8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Trap signals to cleanly kill backend on exit
cleanup() {
    echo ""
    echo "[*] Stopping AI Skin backend (PID $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give backend a moment to boot
sleep 2

# Start Frontend
echo "[*] Starting Vite Frontend on http://localhost:5173..."
cd frontend
npm run dev
