#!/usr/bin/env python3
"""
將 positive/ 與 negative/ 分割成訓練集與驗證集

使用方式（從專案根目錄執行）：
    python tools/prepare_data.py
    python tools/prepare_data.py --val-ratio 0.15

執行後會建立：
    data/train/positive/   訓練用正樣本
    data/train/negative/   訓練用負樣本
    data/val/positive/     驗證用正樣本
    data/val/negative/     驗證用負樣本
"""

import argparse
import random
import shutil
from pathlib import Path

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="將正負樣本分割為訓練集與驗證集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--positive-dir", default="positive",
        help="正樣本資料夾（交界點在圖片中心的 161×161 patch）",
    )
    parser.add_argument(
        "--negative-dir", default="negative",
        help="負樣本資料夾（無交界點或交界點不在中心的 161×161 patch）",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="輸出資料夾（會建立 train/val 子結構）",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="驗證集比例（0.1～0.3，建議 0.2 即 20%%）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="隨機種子（固定數值可重現相同分割結果）",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> list[Path]:
    """收集資料夾中所有支援格式的圖片。"""
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def split_and_copy(
    files: list[Path],
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
) -> tuple[int, int]:
    """隨機分割並複製圖片到 train/val 資料夾，回傳 (train 數, val 數)。"""
    n_val = max(1, int(len(files) * val_ratio))
    val_indices = set(random.sample(range(len(files)), n_val))

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(files):
        dst_dir = val_dir if i in val_indices else train_dir
        shutil.copy2(src, dst_dir / src.name)

    return len(files) - n_val, n_val


def main():
    args = parse_args()
    random.seed(args.seed)

    pos_dir = Path(args.positive_dir)
    neg_dir = Path(args.negative_dir)
    out_dir = Path(args.output_dir)

    for d in (pos_dir, neg_dir):
        if not d.exists():
            raise SystemExit(f"找不到資料夾：{d}")

    pos_files = collect_images(pos_dir)
    neg_files = collect_images(neg_dir)

    if not pos_files:
        raise SystemExit(f"在 {pos_dir} 找不到圖片")
    if not neg_files:
        raise SystemExit(f"在 {neg_dir} 找不到圖片")

    print("分割資料集...")
    n_pos_train, n_pos_val = split_and_copy(
        pos_files,
        out_dir / "train" / "positive",
        out_dir / "val"   / "positive",
        args.val_ratio,
    )
    print(f"  正樣本（共 {len(pos_files)} 張）→ 訓練：{n_pos_train}  驗證：{n_pos_val}")

    n_neg_train, n_neg_val = split_and_copy(
        neg_files,
        out_dir / "train" / "negative",
        out_dir / "val"   / "negative",
        args.val_ratio,
    )
    print(f"  負樣本（共 {len(neg_files)} 張）→ 訓練：{n_neg_train}  驗證：{n_neg_val}")

    print(f"\n完成。資料結構已建立於 {out_dir}/")
    print(f"  train/positive/  {n_pos_train} 張")
    print(f"  train/negative/  {n_neg_train} 張")
    print(f"  val/positive/    {n_pos_val} 張")
    print(f"  val/negative/    {n_neg_val} 張")


if __name__ == "__main__":
    main()
