#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  AI Skin: Intelligent Skin Diseases Detection Setup"
echo "  Target Platform: macOS (Apple Silicon / Intel)"
echo "=========================================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Locate Python 3.11 or Python 3.10+
PYTHON_BIN=""
if command -v /opt/homebrew/bin/python3.11 &>/dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.11"
elif command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "[!] Error: python3 not found. Please install Python 3.10+ (Homebrew: brew install python@3.11)"
    exit 1
fi

echo "[*] Using Python: $($PYTHON_BIN --version) at $PYTHON_BIN"

# Create Virtual Environment if not exists
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment 'venv'..."
    "$PYTHON_BIN" -m venv venv
else
    echo "[*] Virtual environment 'venv' already exists."
fi

# Activate Virtual Environment
source venv/bin/activate
echo "[*] Activated virtual environment: $(which python)"

# Upgrade pip, setuptools, wheel
echo "[*] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify MPS / PyTorch device
echo "[*] Verifying PyTorch hardware acceleration..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f'Apple Silicon MPS Acceleration Available: {mps_avail}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Acceleration Available: {cuda_avail}')
print(f'Default Selected Device: {\"mps\" if mps_avail else (\"cuda\" if cuda_avail else \"cpu\")}')
"

echo "=========================================================="
echo "  Python Environment Setup Complete!"
echo "  Activate with: source venv/bin/activate"
echo "=========================================================="
