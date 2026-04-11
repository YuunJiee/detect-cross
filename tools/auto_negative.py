#!/usr/bin/env python3
"""
用現有模型自動從裁切的 tile 中篩選出負樣本

原理：
    訓練好的模型能辨識「交界點在圖片中心」的 patch。
    對大圖裁切出的 tile 進行預測，
    - 信心高（≥ threshold）→ 交界在中心 → 排除（這是正樣本特徵）
    - 信心低（< threshold）→ 非中心或無交界 → 保留為負樣本

使用方式（從專案根目錄執行）：
    # Step 1：先用 grid_crop.py 裁切大圖
    python tools/grid_crop.py --input 原始照片 --output new_tiles --size 161 --stride 40

    # Step 2：用模型篩出負樣本
    python tools/auto_negative.py --input new_tiles --output new_negative

    # Step 3：合併到 negative/ 後重新訓練
    cp new_negative/* negative/
    python tools/prepare_data.py
    python train.py
"""

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="用模型自動從 tile 中篩選負樣本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="裁切好的 tile 資料夾（由 tools/grid_crop.py 產生）",
    )
    parser.add_argument(
        "--output", required=True,
        help="篩選後的負樣本輸出資料夾",
    )
    parser.add_argument(
        "--model", default="models/cross_detector_v1.pth",
        help="用來篩選的模型路徑",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help=(
            "排除門檻：模型信心 ≥ 此值的 tile 視為「交界在中心」→ 排除。\n"
            "建議設低（0.5）以確保負樣本乾淨，寧可多排除幾張。"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="批次大小（GPU 記憶體不足時調低）",
    )
    return parser.parse_args()


def load_model(model_path: str, device: str):
    """載入模型。"""
    checkpoint = torch.load(model_path, map_location=device)
    pos_idx = checkpoint.get("pos_idx", 1)
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().to(device), pos_idx


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"找不到模型：{model_path}\n請先執行：python train.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置：{device}")

    model, pos_idx = load_model(str(model_path), device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"在 {input_dir} 找不到圖片")

    print(f"共 {len(files)} 張 tile，排除門檻={args.threshold}\n")

    kept = discarded = 0

    for i in tqdm(range(0, len(files), args.batch_size), desc="篩選中"):
        batch_files = files[i : i + args.batch_size]
        tensors = [
            transform(Image.open(f).convert("RGB"))
            for f in batch_files
        ]
        inp = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[:, pos_idx].cpu().tolist()

        for f, prob in zip(batch_files, probs):
            if prob >= args.threshold:
                discarded += 1  # 交界在中心 → 排除
            else:
                shutil.copy2(f, output_dir / f.name)  # 保留為負樣本
                kept += 1

    print(f"\n完成！")
    print(f"  保留為負樣本：{kept} 張 → {output_dir}/")
    print(f"  排除（正樣本特徵）：{discarded} 張")
    print(f"\n下一步：")
    print(f"  cp {output_dir}/* negative/")
    print(f"  python tools/prepare_data.py")
    print(f"  python train.py")


if __name__ == "__main__":
    main()
