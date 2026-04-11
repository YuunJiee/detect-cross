# 訓練與資料前處理指南

本文件給需要「重訓模型」或「重建資料」的使用者。

---

## A. 資料已建立好：直接訓練

```bash
source venv/bin/activate
python train.py
```

訓練完成後推論：

```bash
python inference.py --image original_image/UA00001.bmp
```

常用訓練參數：

```bash
python train.py \
    --epochs 20 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --model-name cross_detector_v1
```

---

## B. 需要重建資料：前處理流程

### B1) 切 tile

```bash
python tools/grid_crop.py \
    --input 原始照片 \
    --output new_tiles \
    --size 161 \
    --stride 40
```

### B2) 自動篩選負樣本

```bash
python tools/auto_negative.py \
    --input new_tiles \
    --output negative \
    --model models/cross_detector_v1.pth \
    --threshold 0.5
```

### B3) 建立 train/val

```bash
python tools/prepare_data.py
```

---

## C. 手動審閱資料（可選）

```bash
python tools/review_and_delete.py --input positive/
```

按鍵：
- `Space` / `→` / `Enter`：保留下一張
- `D` / `Delete` / `BackSpace`：刪除下一張
- `U`：復原上次刪除
- `Q` / `Esc`：離開
