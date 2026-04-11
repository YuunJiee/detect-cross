# 快速推論指南

本文件給只需要「直接跑結果」的使用者。

---

## 1) 一次性安裝

```bash
bash setup.sh
```

> macOS 若沒有 `python`，請改用 `python3`。

---

## 2) 每次使用前

```bash
source venv/bin/activate
```

---

## 3) 單張圖片推論

```bash
python inference.py --image original_image/UA00001.bmp
```

輸出：
- `results/UA00001_result.png`（標註圖）
- `results/UA00001_points.csv`（座標）

---

## 4) 批次推論整個資料夾

```bash
python inference.py --all --photo-dir original_image
```

---

## 5) 常調參數

```bash
python inference.py \
    --image original_image/UA00001.bmp \
    --threshold 0.95 \
    --stride 10 \
    --nms-dist 50
```

- `--threshold`：越高越嚴格（建議 0.90~0.99）
- `--stride`：越小越準但越慢（建議 5~20）
- `--nms-dist`：重複點過濾距離

---

## 6) macOS 補充

- Apple Silicon（M1/M2/M3）會自動使用 MPS（Metal）加速。
- 若使用手動審圖工具時遇到 Tkinter 問題：

```bash
brew install python-tk
```
