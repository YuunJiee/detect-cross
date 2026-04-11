#!/usr/bin/env python3
"""
訓練十字交界點偵測模型（MobileNetV3-Small 二元分類器）

使用方式：
    python train.py
    python train.py --epochs 30 --model-name cross_detector_v2

前置條件：
    先執行 python tools/prepare_data.py 建立 data/ 資料夾
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ─── 參數設定 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="訓練十字交界點 CNN 分類器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="訓練資料根目錄（由 tools/prepare_data.py 建立）",
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="模型儲存資料夾",
    )
    parser.add_argument(
        "--model-name", default="cross_detector_v1",
        help="模型檔案名稱（不含 .pth）",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="訓練回合數（建議 20～30）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="每批次樣本數（GPU 記憶體不足時調低）",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="學習率（建議保持預設）",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="資料讀取執行緒數",
    )
    return parser.parse_args()


# ─── 資料載入 ─────────────────────────────────────────────────────────────────

def make_data_loaders(data_dir: str, batch_size: int, workers: int):
    """建立訓練與驗證資料載入器，含資料增強。"""

    # 訓練集：加入隨機翻轉、旋轉、顏色抖動以增強泛化能力
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 驗證集：不做增強，僅縮放與標準化
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


# ─── 模型建立 ─────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """載入預訓練 MobileNetV3-Small，修改輸出層為二元分類。"""
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)  # 0=negative, 1=positive
    return model


# ─── 損失函數（補償樣本不平衡） ───────────────────────────────────────────────

def make_criterion(dataset, device: str) -> nn.CrossEntropyLoss:
    """
    計算類別權重以補償正負樣本數量不平衡。
    當負樣本遠多於正樣本時，正樣本的 loss 自動加大，避免模型偏向負樣本。
    """
    class_counts = [0, 0]
    for _, label in dataset.samples:
        class_counts[label] += 1

    total = sum(class_counts)
    weights = torch.tensor(
        [total / (2 * c) for c in class_counts],
        dtype=torch.float,
    ).to(device)

    print(f"  正樣本：{class_counts[1]}  負樣本：{class_counts[0]}")
    print(f"  損失權重 → 正：{weights[1]:.2f}  負：{weights[0]:.2f}")
    return nn.CrossEntropyLoss(weight=weights)


# ─── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 確認資料夾存在
    if not Path(f"{args.data_dir}/train").exists():
        raise SystemExit(
            "找不到訓練資料，請先執行：\n"
            "  python tools/prepare_data.py"
        )

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_dir) / f"{args.model_name}.pth"

    # 選擇運算裝置（優先 CUDA，其次 Apple Silicon MPS）
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"使用裝置：{device}")
    if device == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("GPU：Apple Metal (MPS)")

    # 資料
    print("\n載入資料...")
    train_loader, val_loader, train_dataset = make_data_loaders(
        args.data_dir, args.batch_size, args.workers
    )
    class_to_idx = train_dataset.class_to_idx
    pos_idx = class_to_idx.get("positive", 1)
    print(f"類別對應：{class_to_idx}")
    print(f"訓練集：{len(train_dataset)} 張  驗證集：{len(val_loader.dataset)} 張")

    # 模型
    print("\n建立模型...")
    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("計算樣本權重...")
    criterion = make_criterion(train_dataset, device)

    # 訓練
    print(f"\n開始訓練（共 {args.epochs} 個回合）\n{'─'*60}")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # 訓練階段
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)

        # 驗證階段
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        val_acc = correct / total
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"({elapsed:.0f}s)")

        # 儲存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict":   model.state_dict(),
                "class_to_idx": class_to_idx,
                "pos_idx":      pos_idx,
                "val_acc":      best_acc,
                "epochs":       epoch,
            }, model_path)
            print(f"           ✓ 儲存模型（val_acc={best_acc:.4f}）→ {model_path}")

    print(f"\n{'─'*60}")
    print(f"訓練完成。最佳驗證準確率：{best_acc:.4f}")
    print(f"模型路徑：{model_path}")


if __name__ == "__main__":
    main()
