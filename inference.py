#!/usr/bin/env python3
"""
在原始大圖上找出所有十字交界點（滑窗推論）

使用方式：
    # 單張圖片
    python inference.py --image 原始照片/UA00001.bmp

    # 批次處理整個資料夾
    python inference.py --all --photo-dir 原始照片

    # 調整參數
    python inference.py --image 原始照片/UA00001.bmp --threshold 0.95 --stride 5
"""

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# 正負樣本的訓練 patch 尺寸（必須與訓練時一致）
PATCH_SIZE = 161

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ─── 參數設定 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="在原始照片上滑窗偵測十字交界點",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 輸入來源（擇一）
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image",    help="單張圖片路徑")
    source.add_argument("--all",      action="store_true", help="批次處理 --photo-dir 內所有圖片")

    parser.add_argument(
        "--photo-dir", default="原始照片",
        help="批次模式下的圖片資料夾",
    )
    parser.add_argument(
        "--model", default="models/cross_detector_v1.pth",
        help="模型檔案路徑",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="結果輸出資料夾",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95,
        help=(
            "信心門檻（0～1）：模型對某個位置是交界點的把握程度。\n"
            "越高 → 越嚴格，假陽性少但可能漏偵測；越低 → 較寬鬆，偵測較多但可能有誤判。\n"
            "建議範圍：0.90～0.99"
        ),
    )
    parser.add_argument(
        "--stride", type=int, default=10,
        help=(
            "滑窗步長（像素）：每次視窗移動的距離。\n"
            "越小 → 搜尋越細緻、定位越準確，但速度較慢；越大 → 速度快但可能漏掉交界點。\n"
            "建議範圍：5～20"
        ),
    )
    parser.add_argument(
        "--nms-dist", type=int, default=50,
        help=(
            "重複偵測過濾距離（像素）：距離小於此值的重複交界點只保留信心最高的一個。\n"
            "建議設為格子間距的 1/3～1/2。"
        ),
    )
    return parser.parse_args()


# ─── 模型載入 ─────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    """從 .pth 檔案載入模型權重。"""
    checkpoint = torch.load(model_path, map_location=device)
    pos_idx = checkpoint.get("pos_idx", 1)

    model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    val_acc = checkpoint.get("val_acc", "N/A")
    print(f"模型載入：{model_path}  (val_acc={val_acc:.4f})" if isinstance(val_acc, float)
          else f"模型載入：{model_path}")
    return model, pos_idx


def make_transform():
    """建立與訓練時相同的影像前處理流程。"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── 滑窗推論 ─────────────────────────────────────────────────────────────────

def sliding_window_inference(
    img_bgr,
    model,
    transform,
    pos_idx: int,
    stride: int,
    threshold: float,
    device: str,
) -> list[tuple[int, int, float]]:
    """
    以固定步長在整張圖片上滑動 PATCH_SIZE×PATCH_SIZE 的視窗，
    對每個視窗用模型判斷是否為交界點。

    交界點座標 = 視窗左上角 + (PATCH_SIZE//2, PATCH_SIZE//2)
    因為訓練樣本的交界點永遠位於 patch 正中心。

    回傳：超過 threshold 的候選點列表 [(cx, cy, prob), ...]
    """
    h, w = img_bgr.shape[:2]
    candidates = []
    batch_tensors = []
    batch_coords  = []
    BATCH_SIZE = 64  # 一次送入模型的 patch 數（可依 GPU 記憶體調整）

    def flush_batch():
        """將累積的 patch 批次送入模型推論。"""
        if not batch_tensors:
            return
        inp = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[:, pos_idx].cpu().tolist()
        for (cx, cy), prob in zip(batch_coords, probs):
            if prob >= threshold:
                candidates.append((cx, cy, prob))
        batch_tensors.clear()
        batch_coords.clear()

    for y in range(0, h - PATCH_SIZE + 1, stride):
        for x in range(0, w - PATCH_SIZE + 1, stride):
            patch_bgr = img_bgr[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            batch_tensors.append(transform(Image.fromarray(patch_rgb)))
            batch_coords.append((x + PATCH_SIZE // 2, y + PATCH_SIZE // 2))

            if len(batch_tensors) == BATCH_SIZE:
                flush_batch()

    flush_batch()
    return candidates


# ─── NMS 後處理 ───────────────────────────────────────────────────────────────

def non_maximum_suppression(
    candidates: list[tuple[int, int, float]],
    min_dist: int,
) -> list[tuple[int, int, float]]:
    """
    非最大值抑制（NMS）：同一個交界點因視窗重疊可能產生多個候選，
    保留信心最高的，刪除距離小於 min_dist 像素的其他候選。
    """
    candidates = sorted(candidates, key=lambda p: -p[2])  # 信心由高到低排序
    kept = []
    for cx, cy, prob in candidates:
        too_close = any(
            ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5 < min_dist
            for fx, fy, _ in kept
        )
        if not too_close:
            kept.append((cx, cy, prob))
    return kept


# ─── 結果視覺化 ───────────────────────────────────────────────────────────────

def draw_results(img_bgr, points: list[tuple[int, int, float]]) -> None:
    """在圖片上標記交界點（紅色實心圓 + 黃色輪廓 + 綠色信心值）。"""
    for cx, cy, prob in points:
        cv2.circle(img_bgr, (cx, cy), 8,  (0, 0, 255),   -1)
        cv2.circle(img_bgr, (cx, cy), 10, (0, 255, 255),  2)
        cv2.putText(
            img_bgr, f"{prob:.2f}", (cx + 12, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
        )


# ─── 單張圖片處理 ─────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    model,
    transform,
    pos_idx: int,
    args,
    out_dir: Path,
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[警告] 無法開啟：{img_path}")
        return

    h, w = img.shape[:2]
    t0 = time.time()
    print(f"\n處理：{img_path.name}  ({w}×{h}px)")

    device = next(model.parameters()).device.type
    candidates = sliding_window_inference(
        img, model, transform, pos_idx, args.stride, args.threshold, device
    )
    points = non_maximum_suppression(candidates, args.nms_dist)

    elapsed = time.time() - t0
    print(f"  候選點：{len(candidates)}  →  NMS 後：{len(points)}  ({elapsed:.1f}s)")

    for i, (cx, cy, prob) in enumerate(points, 1):
        print(f"  [{i:3d}]  cx={cx:5d}  cy={cy:5d}  信心={prob:.3f}")

    # 儲存標記圖
    result_img = img.copy()
    draw_results(result_img, points)
    out_path = out_dir / f"{img_path.stem}_result.png"
    cv2.imwrite(str(out_path), result_img)

    # 儲存座標 CSV
    csv_path = out_dir / f"{img_path.stem}_points.csv"
    with open(csv_path, "w") as f:
        f.write("id,cx,cy,confidence\n")
        for i, (cx, cy, prob) in enumerate(
            sorted(points, key=lambda p: (p[1], p[0])), 1
        ):
            f.write(f"{i},{cx},{cy},{prob:.4f}\n")

    print(f"  結果圖：{out_path}")
    print(f"  座標表：{csv_path}")


# ─── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"找不到模型：{model_path}\n"
            "請先執行：python train.py"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用裝置：{device}")

    model, pos_idx = load_model(str(model_path), device)
    transform = make_transform()

    print(f"\n推論設定")
    print(f"  信心門檻：{args.threshold}")
    print(f"  滑窗步長：{args.stride} px")
    print(f"  NMS 距離：{args.nms_dist} px")

    if args.all:
        photo_dir = Path(args.photo_dir)
        images = sorted(
            p for p in photo_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise SystemExit(f"在 {photo_dir} 找不到圖片")
        print(f"\n批次處理：{len(images)} 張")
        for img_path in images:
            process_image(img_path, model, transform, pos_idx, args, out_dir)
    else:
        process_image(Path(args.image), model, transform, pos_idx, args, out_dir)

    print(f"\n完成。所有結果存於：{out_dir}/")


if __name__ == "__main__":
    main()
