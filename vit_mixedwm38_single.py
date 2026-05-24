
import os
import csv
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm


# MixedWM38 single-defect subset:
# 0 = Normal, 1~8 = single defect classes from the 8-dimensional one-hot label.
CLASS_NAMES = [
    "Normal",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_single_defect_subset(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    讀取 MixedWM38 .npz，並篩選 single-defect subset。

    預期格式：
      arr_0: wafer map images, shape [N, 52, 52]，像素值 0/1/2
      arr_1: defect labels, shape [N, 8]，8 種基本缺陷的 one-hot/multi-hot label

    篩選規則：
      label sum = 0 -> Normal
      label sum = 1 -> Single defect
      label sum > 1 -> Mixed defect，排除不用
    """
    data = np.load(npz_path, allow_pickle=True)

    if "arr_0" not in data or "arr_1" not in data:
        raise KeyError(
            f"找不到 arr_0 或 arr_1。此 npz 內含 keys={list(data.keys())}。\n"
            "請確認你下載的是 MixedWM38.npz。"
        )

    x = np.asarray(data["arr_0"])
    y = np.asarray(data["arr_1"])

    # 移除多餘 channel，例如 [N, 52, 52, 1] -> [N, 52, 52]
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"arr_0 應為 [N,H,W]，但目前 shape={x.shape}")

    if y.ndim != 2 or y.shape[1] != 8:
        raise ValueError(f"arr_1 應為 [N,8] one-hot/multi-hot label，但目前 shape={y.shape}")

    y = y.astype(np.int64)
    label_sum = y.sum(axis=1)

    # 只保留 Normal 與 Single-defect，排除 mixed defect
    keep = (label_sum == 0) | (label_sum == 1)
    x_single = x[keep]
    y_single_hot = y[keep]
    label_sum_single = label_sum[keep]

    # Normal -> 0；single defect -> argmax + 1
    y_single = np.zeros(len(y_single_hot), dtype=np.int64)
    defect_mask = label_sum_single == 1
    y_single[defect_mask] = np.argmax(y_single_hot[defect_mask], axis=1) + 1

    return x_single.astype(np.int64), y_single


class WaferMapDataset(Dataset):
    """
    Wafer map 影像資料集。

    重要設計：
      - MixedWM38 wafer map 的值是 0, 1, 2。
      - 這不是一般灰階影像，而是類別圖：
          0: blank spot
          1: normal die
          2: failed/broken die
      - 因此這裡轉成 3-channel one-hot image，比單純除以 2 更合理。
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_size: int = 64,
        augment: bool = False,
    ):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W]
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical flip

        # 旋轉 0/90/180/270 度。
        # 對 wafer defect pattern 通常合理，因為類別關心的是形狀型態而非絕對方向。
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            x = torch.rot90(x, k=k, dims=[1, 2])
        return x

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).long()  # [H,W], values 0/1/2
        img = torch.clamp(img, 0, 2)

        # [H,W] -> [H,W,3] -> [3,H,W]
        x = F.one_hot(img, num_classes=3).permute(2, 0, 1).float()

        # resize 到 ViT 方便處理的大小，例如 64x64。
        # one-hot 類別圖用 nearest 比 bilinear 更適合。
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        if self.augment:
            x = self._augment(x)

        y = int(self.labels[idx])
        return x, y


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必須可以被 patch_size 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: [B,C,H,W]
        x = self.proj(x)                  # [B,embed_dim,H/P,W/P]
        x = x.flatten(2).transpose(1, 2)  # [B,num_patches,embed_dim]
        return x


class SimpleViT(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 9,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # class token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)  # [B,N,D]

        cls = self.cls_token.expand(b, -1, -1)  # [B,1,D]
        x = torch.cat([cls, x], dim=1)          # [B,N+1,D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = self.norm(x[:, 0])  # 取 class token
        logits = self.head(x)
        return logits


def make_loaders(
    images: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_weighted_sampler: bool,
):
    # 先切出 test，再從 train_val 切出 validation
    idx = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=seed,
        stratify=labels[train_val_idx],
    )

    train_set = WaferMapDataset(images[train_idx], labels[train_idx], image_size=image_size, augment=True)
    val_set = WaferMapDataset(images[val_idx], labels[val_idx], image_size=image_size, augment=False)
    test_set = WaferMapDataset(images[test_idx], labels[test_idx], image_size=image_size, augment=False)

    if use_weighted_sampler:
        train_labels = labels[train_idx]
        class_count = np.bincount(train_labels, minlength=len(CLASS_NAMES))
        class_weight = 1.0 / np.maximum(class_count, 1)
        sample_weight = class_weight[train_labels]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weight),
            num_samples=len(sample_weight),
            replacement=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    split_info = {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }
    return train_loader, val_loader, test_loader, split_info


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

        pbar.set_postfix(loss=float(loss.item()))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    return metrics, all_targets, all_preds, all_probs


def plot_history(history: List[Dict[str, float]], output_dir: str):
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train Accuracy")
    plt.plot(epochs, [h["val_acc"] for h in history], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=200)
    plt.close()


def save_history_csv(history: List[Dict[str, float]], output_dir: str):
    path = os.path.join(output_dir, "history.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(11, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


@torch.no_grad()
def plot_misclassified_examples(model, loader, device, output_dir: str, max_examples: int = 16):
    model.eval()
    examples = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        x_cpu = x.cpu()

        for i in range(x_cpu.size(0)):
            if preds[i].item() != y[i].item():
                examples.append((x_cpu[i], int(y[i].item()), int(preds[i].item())))
                if len(examples) >= max_examples:
                    break
        if len(examples) >= max_examples:
            break

    if len(examples) == 0:
        print("沒有找到誤分類樣本，因此不產生 misclassified_examples.png")
        return

    cols = 4
    rows = math.ceil(len(examples) / cols)
    plt.figure(figsize=(cols * 3.2, rows * 3.2))

    for idx, (x, true_label, pred_label) in enumerate(examples):
        # 將 one-hot 3-channel 還原成 0/1/2 wafer map 方便顯示
        wafer = torch.argmax(x, dim=0).numpy()

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(wafer, interpolation="nearest")
        plt.axis("off")
        plt.title(
            f"T: {CLASS_NAMES[true_label]}\nP: {CLASS_NAMES[pred_label]}",
            fontsize=9,
        )

    plt.suptitle("Misclassified Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "misclassified_examples.png"), dpi=200)
    plt.close()


def save_class_distribution(labels: np.ndarray, output_dir: str):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))

    with open(os.path.join(output_dir, "class_distribution.txt"), "w", encoding="utf-8") as f:
        for name, count in zip(CLASS_NAMES, counts):
            f.write(f"{name}: {int(count)}\n")

    plt.figure(figsize=(10, 5))
    plt.bar(CLASS_NAMES, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Samples")
    plt.title("Single-defect Subset Class Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="ViT for MixedWM38 single-defect wafer classification")

    parser.add_argument("--npz_path", type=str, required=True, help="MixedWM38.npz 路徑")
    parser.add_argument("--output_dir", type=str, default="./runs/vit_mixedwm38_single")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--weighted_sampler", action="store_true", help="使用 WeightedRandomSampler 處理類別不平衡")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    print("讀取資料並篩選 single-defect subset...")
    images, labels = load_single_defect_subset(args.npz_path)
    print(f"Single-defect subset 總數: {len(labels)}")
    save_class_distribution(labels, args.output_dir)

    train_loader, val_loader, test_loader, split_info = make_loaders(
        images=images,
        labels=labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=args.weighted_sampler,
    )
    print(f"資料切分: {split_info}")

    model = SimpleViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=len(CLASS_NAMES),
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_val_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_vit_mixedwm38_single.pth")
    no_improve = 0
    history = []

    print("開始訓練...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        # 以 macro F1 作為 best model 選擇依據，較適合多類別且可能不平衡的資料
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "class_names": CLASS_NAMES,
                    "best_val_f1": float(best_val_f1),
                },
                best_path,
            )
            print(f"  -> 儲存最佳模型: {best_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping: validation F1 已連續 {args.patience} epochs 沒改善")
            break

    save_history_csv(history, args.output_dir)
    plot_history(history, args.output_dir)

    print("載入最佳模型並在 test set 評估...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, y_true, y_pred, _ = evaluate(model, test_loader, criterion, device)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\nTest Metrics")
    print(test_metrics)
    print("\nClassification Report")
    print(report)

    with open(os.path.join(args.output_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write("Test Metrics\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nClassification Report\n")
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, args.output_dir)
    plot_misclassified_examples(model, test_loader, device, args.output_dir, max_examples=16)

    print("\n完成！輸出檔案位於：", args.output_dir)
    print("主要輸出：")
    print("  - best_vit_mixedwm38_single.pth")
    print("  - history.csv")
    print("  - loss_curve.png")
    print("  - accuracy_curve.png")
    print("  - confusion_matrix.png")
    print("  - misclassified_examples.png")
    print("  - test_report.txt")
    print("  - class_distribution.png")


if __name__ == "__main__":
    main()

"""
python vit_mixedwm38_single.py \
  --npz_path ./datasets/MixedWM38.npz \
  --output_dir ./runs/vit_mixedwm38_single \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-3 \
  --weighted_sampler

"""