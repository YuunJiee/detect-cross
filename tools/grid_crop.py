#!/usr/bin/env python3
"""
將大圖裁切成固定大小的方形 tile

主要用途：從原始大圖產生負樣本候選，再透過 auto_negative.py 篩選。

使用方式（從專案根目錄執行）：
    # 產生用於篩選負樣本的 tile（stride=40 讓 tile 有重疊，涵蓋更多偏移情況）
    python tools/grid_crop.py --input 原始照片 --output new_tiles --size 161 --stride 40

    # 不重疊裁切（每格只切一次）
    python tools/grid_crop.py --input 原始照片 --output new_tiles --size 161
"""

import argparse
from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="將資料夾內所有圖片裁切成固定大小的方形 tile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="原始圖片資料夾",
    )
    parser.add_argument(
        "--output", required=True,
        help="裁切後 tile 的輸出資料夾",
    )
    parser.add_argument(
        "--size", type=int, default=161,
        help="tile 的邊長（像素），必須與正樣本尺寸相同",
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help=(
            "相鄰 tile 的間距（像素）。\n"
            "預設等於 --size（不重疊）；\n"
            "設為較小值（如 40）可產生有重疊的 tile，用於負樣本增強。"
        ),
    )
    parser.add_argument(
        "--ext", default=".bmp",
        help="輸出圖片格式（.bmp / .png / .jpg）",
    )
    return parser.parse_args()


def get_crop_positions(length: int, tile: int, stride: int) -> list[int]:
    """計算裁切起始座標列表。"""
    if length < tile:
        return []
    return list(range(0, length - tile + 1, stride))


def crop_image(image_path: Path, out_dir: Path, tile_size: int,
               stride: int, out_ext: str) -> int:
    """裁切單張圖片，回傳產生的 tile 數量。"""
    with Image.open(image_path) as img:
        width, height = img.size
        xs = get_crop_positions(width,  tile_size, stride)
        ys = get_crop_positions(height, tile_size, stride)

        count = 0
        stem = image_path.stem
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                tile.save(out_dir / f"{stem}_r{r:03d}_c{c:03d}{out_ext}")
                count += 1
    return count


def main():
    args = parse_args()

    input_dir  = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"找不到資料夾：{input_dir}")
    if args.size <= 0:
        raise SystemExit("--size 必須 > 0")

    stride = args.stride if args.stride is not None else args.size
    if stride <= 0:
        raise SystemExit("--stride 必須 > 0")

    out_ext = args.ext.lower()
    if not out_ext.startswith("."):
        out_ext = f".{out_ext}"

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"在 {input_dir} 找不到圖片")

    print(f"裁切設定：size={args.size}  stride={stride}  格式={out_ext}")
    print(f"圖片數量：{len(images)} 張\n")

    total_tiles = 0
    for img_path in images:
        count = crop_image(img_path, output_dir, args.size, stride, out_ext)
        print(f"  {img_path.name} → {count} tiles")
        total_tiles += count

    print(f"\n完成。共產生 {total_tiles} 個 tile → {output_dir}/")


if __name__ == "__main__":
    main()
