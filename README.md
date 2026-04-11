# 十字交界點自動偵測系統

利用深度學習（CNN）在航拍照片中自動標記白色粉筆格線的交界點。

---

## 資料夾結構

```
detectCross/
├── positive/               ← 正樣本（已提供，交界點在圖片中心）
├── negative/               ← 負樣本（需建立，見 STEP 1）
├── original_image/         ← 要偵測的大圖
├── models/                 ← 訓練好的模型
│   └── cross_detector_v1.pth
├── data/                   ← 訓練/驗證資料（由 tools/prepare_data.py 自動建立）
├── results/                ← 推論結果輸出
├── tools/                  ← 資料處理工具
│   ├── grid_crop.py        裁切大圖為小 tile
│   ├── auto_negative.py    用模型自動篩選負樣本
│   ├── prepare_data.py     建立訓練/驗證資料集
│   └── review_and_delete.py  手動審閱圖片的 GUI 工具
├── train.py                ← 訓練模型
├── inference.py            ← 推論（偵測交界點）
├── setup.sh                ← 一鍵安裝環境
└── README.md               ← 本說明文件
```

---

## 系統需求

- Python 3.10 以上
- NVIDIA GPU（選用，無 GPU 也可執行，但速度較慢）
- 作業系統：Linux / macOS / Windows WSL2

---

## 安裝環境（第一次使用）

### macOS 補充

- 建議先安裝 Homebrew 與 Python 3.10+。
- 若 `python` 指令不存在，請改用 `python3`（本專案都可用 `python3` 執行）。
- 若要使用手動審閱工具（Tk GUI），若遇到 Tkinter 缺失可先安裝：

```bash
brew install python-tk
```

在 Apple Silicon（M1/M2/M3）上，程式會自動使用 MPS（Metal）加速；若不可用則退回 CPU。

### Step 0-A：安裝系統套件（Ubuntu / WSL2）

```bash
sudo apt install python3.12-venv -y
```

### Step 0-B：執行安裝腳本

```bash
bash setup.sh
```

腳本會自動：
- 建立 Python 虛擬環境（`venv/`）
- 偵測是否有 NVIDIA GPU，安裝對應版本的 PyTorch
- 安裝所有必要套件（Pillow、OpenCV、tqdm 等）

### Step 0-C：每次使用前啟動虛擬環境

```bash
source venv/bin/activate
```

> 啟動後命令列前方會出現 `(venv)` 字樣，代表環境已啟動。

---

## 快速開始

請選擇對應說明文件：

1. 只要推論：[docs/TEACHER_INFERENCE_GUIDE.md](docs/TEACHER_INFERENCE_GUIDE.md)
2. 需要重訓或重建資料：[docs/TRAINING_AND_PREPROCESS_GUIDE.md](docs/TRAINING_AND_PREPROCESS_GUIDE.md)

---

## 最短指令

```bash
bash setup.sh
source venv/bin/activate
python inference.py --image original_image/UA00001.bmp
```

---

## 其他工具

- 手動審圖刪圖工具：`python tools/review_and_delete.py --input positive/`
- 網格擬合工具（進階）：`python fit_grid.py --image 原始照片/UA00001.bmp`
