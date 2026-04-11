#!/bin/bash
# 建立 Python 虛擬環境並安裝所有必要套件
# 執行方式：bash setup.sh

set -e  # 遇到錯誤立即停止

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================="
echo "  十字交界點偵測 - 環境安裝"
echo "=============================="
echo ""

# ── 建立虛擬環境 ──────────────────────────────────────────────────────────────
echo "[1/4] 建立虛擬環境..."
python3 -m venv venv
source venv/bin/activate

# ── 升級 pip ─────────────────────────────────────────────────────────────────
echo "[2/4] 升級套件管理器..."
pip install --upgrade pip -q

# ── 安裝 PyTorch（自動偵測 GPU）────────────────────────────────────────────────
echo "[3/4] 安裝 PyTorch..."
if python3 -c "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if r.returncode==0 else 1)" 2>/dev/null; then
    echo "      偵測到 NVIDIA GPU，安裝 CUDA 版本（較大，約需數分鐘）..."
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 -q
    # 修正 WSL2 環境下的函式庫路徑問題
    CUDA_LIB_PATH="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia/cusparselt/lib"
    if [ -d "$CUDA_LIB_PATH" ]; then
        echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH" >> venv/bin/activate
        echo "      已設定 CUDA 函式庫路徑"
    fi
else
    echo "      未偵測到 GPU，安裝 CPU 版本..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── 安裝其他套件 ──────────────────────────────────────────────────────────────
echo "[4/4] 安裝其他必要套件..."
pip install pillow opencv-python-headless tqdm -q

# ── 完成 ─────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "  安裝完成！"
echo "=============================="
echo ""
echo "使用方式："
echo "  1. 啟動環境：source venv/bin/activate"
echo "  2. 建立資料：python tools/prepare_data.py"
echo "  3. 訓練模型：python train.py"
echo "  4. 執行推論：python inference.py --image original_image/UA00001.bmp"
echo ""
echo "詳細說明請參閱 README.md"
