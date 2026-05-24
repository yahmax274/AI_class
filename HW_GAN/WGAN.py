"""
Conditional WGAN-GP for MixedWM38 wafer map generation
=======================================================

功能：
1. 讀取 MixedWM38 .npz 資料集。
2. 使用 Conditional WGAN-GP 訓練 GAN。
3. 根據類別生成 synthetic wafer maps。
4. 以 MSE 評估生成圖與真實圖的差異。
5. 輸出 loss curve、生成圖比較圖、MSE 報表。
6. 可選擇生成補足資料，使各類別數量平衡。

建議資料格式：
.npz 檔中至少需要包含 wafer map 影像與 label。
常見 key 可能是：
- arr_0: wafer maps, shape = [N, H, W] 或 [N, H, W, 1]
- arr_1: labels, shape = [N]

如果你的 key 不是 arr_0 / arr_1，程式會嘗試自動尋找數值陣列。

執行範例：
python conditional_wgan_gp_mixedwm38.py

請先修改 CFG 裡面的 npz_path。
"""

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


# ============================================================
# 1. 參數設定
# ============================================================

@dataclass
class CFG:
    # ---------- 路徑設定 ----------
    npz_path: str = r"./MixedWM38.npz"       # 請改成你的 MixedWM38 .npz 路徑
    output_dir: str = r"./outputs_cwgangp_mixedwm38"

    # ---------- 資料設定 ----------
    image_size: int = 64                     # wafer map 會 resize 成 64x64
    num_classes: int = 38                    # MixedWM38: 1 normal + 8 single + 29 mixed
    use_weighted_sampler: bool = True        # 類別不平衡時建議開啟

    # ---------- 模型設定 ----------
    latent_dim: int = 128                    # noise vector 維度
    label_emb_dim: int = 128                 # label embedding 維度
    base_ch: int = 64                        # CNN 基礎通道數

    # ---------- 訓練設定 ----------
    epochs: int = 100                        # 作業可先用 100；結果不佳可改 200~300
    batch_size: int = 64
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.0                       # WGAN-GP 常用 betas=(0.0, 0.9)
    beta2: float = 0.9
    n_critic: int = 5                        # 每訓練 1 次 G，先訓練 Critic 幾次
    lambda_gp: float = 10.0                  # gradient penalty 權重

    # ---------- 輸出設定 ----------
    sample_every: int = 5                    # 每幾個 epoch 輸出生成圖
    num_vis_per_class: int = 8               # 每個類別視覺化幾張生成圖
    gen_per_class_for_eval: int = 100        # 每類生成幾張用於 MSE 評估
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 平衡資料輸出設定 ----------
    save_balanced_npz: bool = True
    save_generated_images: bool = True       # 是否把生成圖也存成 png


# ============================================================
# 2. 工具函式
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def print_npz_keys(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    print("\n[NPZ keys]")
    for k in data.files:
        arr = data[k]
        print(f"  {k}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
    print()


def auto_load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    自動從 .npz 中找出 wafer maps 與 labels。

    優先順序：
    1. arr_0 當 images，arr_1 當 labels
    2. images / labels
    3. x / y
    4. 自動判斷：維度 >= 3 的數值陣列當 images，維度 1 的數值陣列當 labels
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    # 常見格式 1
    if "arr_0" in keys and "arr_1" in keys:
        images = data["arr_0"]
        labels = data["arr_1"]
        return images, labels

    # 常見格式 2
    if "images" in keys and "labels" in keys:
        images = data["images"]
        labels = data["labels"]
        return images, labels

    # 常見格式 3
    if "x" in keys and "y" in keys:
        images = data["x"]
        labels = data["y"]
        return images, labels

    # 自動猜測
    image_key = None
    label_key = None
    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray):
            if arr.ndim >= 3 and image_key is None:
                image_key = k
            if arr.ndim == 1 and label_key is None:
                label_key = k

    if image_key is None or label_key is None:
        raise ValueError(
            "無法自動判斷 .npz 裡面的 image 與 label。\n"
            "請先用 print_npz_keys(npz_path) 檢查 key，然後手動修改 auto_load_npz()。"
        )

    print(f"[Auto Load] images key = {image_key}, labels key = {label_key}")
    return data[image_key], data[label_key]


def convert_labels_to_int(labels: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    將 labels 轉成 0 ~ C-1 的整數。

    重要：MixedWM38 的 arr_1 通常是 8 維 multi-hot label。
    例如：
    - [0,0,0,0,0,0,0,0] 可能代表 Normal
    - [1,0,0,0,0,0,0,0] 代表某一種 single defect
    - [1,0,1,0,0,0,0,0] 代表 mixed defect

    因此不能直接使用 argmax，否則 38 類會被錯誤壓成 8 類。
    正確做法是：把每一個 unique 8-bit / 8-dim label vector 視為一個類別。
    """
    labels = np.asarray(labels)

    # ------------------------------------------------------------
    # 情況 1：labels 是 2D，例如 [N, 8]
    # 對 MixedWM38，這通常不是 one-hot，而是 multi-hot。
    # 所以每個 unique row 都是一個 defect pattern class。
    # ------------------------------------------------------------
    if labels.ndim == 2:
        # 轉成 tuple，方便當 dict key
        label_tuples = [tuple(row.astype(int).tolist()) for row in labels]

        # 排序方式：先依照缺陷數量排序，再依照 bit pattern 排序。
        # 這樣 normal 通常會在 class 0，single defects 接在後面，mixed defects 再後面。
        unique_labels = sorted(
            list(set(label_tuples)),
            key=lambda x: (sum(x), x)
        )

        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}
        labels_int = np.array([label_to_idx[x] for x in label_tuples], dtype=np.int64)
        return labels_int, idx_to_label

    # ------------------------------------------------------------
    # 情況 2：labels 是 1D，例如 [N]
    # 這種情況才直接把不同 label value 映射成 class index。
    # ------------------------------------------------------------
    if labels.ndim == 1:
        unique_labels = sorted(list(set(labels.tolist())), key=lambda x: str(x))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        idx_to_label = {i: lab for lab, i in label_to_idx.items()}
        labels_int = np.array([label_to_idx[x] for x in labels], dtype=np.int64)
        return labels_int, idx_to_label

    raise ValueError(f"不支援的 labels 維度：{labels.shape}")


def preprocess_images(images: np.ndarray, image_size: int) -> torch.Tensor:
    """
    將 wafer maps 轉成 Tensor，並正規化到 [-1, 1]。

    支援輸入：
    - [N, H, W]
    - [N, H, W, 1]
    - [N, 1, H, W]
    - [N, H, W, 3] 會轉成灰階近似處理
    """
    images = np.asarray(images)

    if images.ndim == 3:
        # [N, H, W] -> [N, 1, H, W]
        images = images[:, None, :, :]
    elif images.ndim == 4:
        # [N, H, W, C] -> [N, C, H, W]
        if images.shape[-1] in [1, 3]:
            images = np.transpose(images, (0, 3, 1, 2))
        # 如果是 RGB，轉成單通道
        if images.shape[1] == 3:
            images = images.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"不支援的 images 維度：{images.shape}")

    images = images.astype(np.float32)

    # 將任意數值範圍縮放到 [0, 1]
    vmin = float(images.min())
    vmax = float(images.max())
    if vmax - vmin < 1e-8:
        raise ValueError("images 的最大值與最小值太接近，無法正規化。")
    images = (images - vmin) / (vmax - vmin)

    x = torch.from_numpy(images).float()

    # resize 到固定大小
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="nearest")

    # [0, 1] -> [-1, 1]
    x = x * 2.0 - 1.0
    return x


def denorm_img(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]，用於視覺化與儲存。"""
    return (x.clamp(-1, 1) + 1.0) / 2.0


# ============================================================
# 3. Dataset
# ============================================================

class WaferMapDataset(Dataset):
    def __init__(self, images_tensor: torch.Tensor, labels: np.ndarray):
        self.images = images_tensor
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_dataloader(dataset: WaferMapDataset, cfg: CFG) -> DataLoader:
    """
    若 use_weighted_sampler=True，會讓少數類別在訓練時被抽到的機率提高。
    這對 MixedWM38 這種不平衡資料集很重要。
    """
    if not cfg.use_weighted_sampler:
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True,
        )

    labels = dataset.labels.numpy()
    class_count = np.bincount(labels)
    class_count[class_count == 0] = 1
    class_weights = 1.0 / class_count
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )


# ============================================================
# 4. Conditional WGAN-GP 模型
# ============================================================

class Generator(nn.Module):
    """
    Conditional Generator

    輸入：
    - z: random noise, shape = [B, latent_dim]
    - y: class label, shape = [B]

    輸出：
    - fake wafer map, shape = [B, 1, 64, 64]
    """
    def __init__(self, latent_dim: int, num_classes: int, label_emb_dim: int, base_ch: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        in_dim = latent_dim + label_emb_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, base_ch * 8 * 4 * 4),
            nn.BatchNorm1d(base_ch * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.net = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(base_ch, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.label_emb(y)
        h = torch.cat([z, y_emb], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), -1, 4, 4)
        out = self.net(h)
        return out


class Critic(nn.Module):
    """
    Conditional Critic

    WGAN-GP 裡面通常稱為 Critic，不稱為 Discriminator。
    因為輸出不是 0~1 的機率，而是一個 real-valued score。

    條件做法：
    - label embedding 轉成 H*W
    - reshape 成 [B, 1, H, W]
    - 與 wafer map concatenate 成 [B, 2, H, W]
    """
    def __init__(self, num_classes: int, image_size: int, base_ch: int):
        super().__init__()
        self.image_size = image_size
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        self.net = nn.Sequential(
            # input: [B, 2, 64, 64]
            nn.Conv2d(2, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64 -> 32 -> 16 -> 8 -> 4
        self.fc = nn.Linear(base_ch * 8 * 4 * 4, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        y_map = self.label_emb(y).view(b, 1, self.image_size, self.image_size)
        h = torch.cat([x, y_map], dim=1)
        h = self.net(h)
        h = h.view(b, -1)
        score = self.fc(h)
        return score.view(-1)


# ============================================================
# 5. WGAN-GP Loss
# ============================================================

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor, device: str) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, labels)

    grad = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad = grad.view(b, -1)
    grad_norm = grad.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


# ============================================================
# 6. 生成與視覺化
# ============================================================

@torch.no_grad()
def generate_by_labels(generator: nn.Module, labels: torch.Tensor, cfg: CFG) -> torch.Tensor:
    generator.eval()
    labels = labels.to(cfg.device)
    z = torch.randn(labels.size(0), cfg.latent_dim, device=cfg.device)
    fake = generator(z, labels)
    return fake


@torch.no_grad()
def save_sample_grid(generator: nn.Module, epoch: int, cfg: CFG):
    generator.eval()
    out_dir = os.path.join(cfg.output_dir, "samples")
    ensure_dir(out_dir)

    labels = []
    for c in range(cfg.num_classes):
        labels += [c] * cfg.num_vis_per_class
    labels = torch.tensor(labels, dtype=torch.long, device=cfg.device)

    fake = generate_by_labels(generator, labels, cfg)
    grid = make_grid(denorm_img(fake), nrow=cfg.num_vis_per_class, padding=2)
    save_path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    save_image(grid, save_path)


def plot_loss_curve(history: Dict[str, List[float]], cfg: CFG):
    save_path = os.path.join(cfg.output_dir, "loss_curve.png")

    plt.figure(figsize=(10, 5))
    plt.plot(history["g_loss"], label="Generator Loss")
    plt.plot(history["d_loss"], label="Critic Loss")
    plt.plot(history["gp"], label="Gradient Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional WGAN-GP Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_real_fake_compare(real_images: torch.Tensor, labels: np.ndarray, generator: nn.Module, cfg: CFG):
    """
    每個類別挑真實圖與生成圖做比較。
    上半部是真實圖，下半部是生成圖。
    """
    out_dir = os.path.join(cfg.output_dir, "compare_real_fake")
    ensure_dir(out_dir)

    generator.eval()
    real_images = real_images.cpu()
    labels_np = np.asarray(labels)

    for c in range(cfg.num_classes):
        idx = np.where(labels_np == c)[0]
        if len(idx) == 0:
            continue

        pick = np.random.choice(idx, size=min(cfg.num_vis_per_class, len(idx)), replace=len(idx) < cfg.num_vis_per_class)
        real = real_images[pick]

        c_labels = torch.full((cfg.num_vis_per_class,), c, dtype=torch.long, device=cfg.device)
        fake = generate_by_labels(generator, c_labels, cfg).cpu()

        both = torch.cat([real[:cfg.num_vis_per_class], fake], dim=0)
        grid = make_grid(denorm_img(both), nrow=cfg.num_vis_per_class, padding=2)
        save_image(grid, os.path.join(out_dir, f"class_{c:02d}_real_top_fake_bottom.png"))


# ============================================================
# 7. MSE 評估
# ============================================================

def compute_class_mean_mse(real_images: torch.Tensor, labels: np.ndarray, generator: nn.Module, cfg: CFG) -> pd.DataFrame:
    """
    方法 1：Generated class mean vs Real class mean

    對每個類別：
    - 取該類所有 real wafer maps 的平均圖
    - 生成多張 fake wafer maps 後取平均圖
    - 計算兩張平均圖的 MSE
    """
    rows = []
    labels_np = np.asarray(labels)
    real_images = real_images.to(cfg.device)

    generator.eval()
    with torch.no_grad():
        for c in range(cfg.num_classes):
            idx = np.where(labels_np == c)[0]
            if len(idx) == 0:
                rows.append({"class_id": c, "num_real": 0, "mean_mse": np.nan})
                continue

            real_c = real_images[idx]
            real_mean = real_c.mean(dim=0, keepdim=True)

            gen_labels = torch.full((cfg.gen_per_class_for_eval,), c, dtype=torch.long, device=cfg.device)
            fake_c = generate_by_labels(generator, gen_labels, cfg)
            fake_mean = fake_c.mean(dim=0, keepdim=True)

            mse = F.mse_loss(fake_mean, real_mean).item()
            rows.append({"class_id": c, "num_real": len(idx), "mean_mse": mse})

    df = pd.DataFrame(rows)
    return df


def compute_nearest_real_mse(real_images: torch.Tensor, labels: np.ndarray, generator: nn.Module, cfg: CFG) -> pd.DataFrame:
    """
    方法 2：Nearest real sample MSE

    對每一張 generated wafer map：
    - 在同類別 real wafer maps 中找 MSE 最小的一張
    - 記錄 nearest MSE

    注意：如果資料量很大，這個計算會比較慢。
    """
    rows = []
    labels_np = np.asarray(labels)
    real_images = real_images.to(cfg.device)

    generator.eval()
    with torch.no_grad():
        for c in range(cfg.num_classes):
            idx = np.where(labels_np == c)[0]
            if len(idx) == 0:
                rows.append({"class_id": c, "num_real": 0, "nearest_mse": np.nan})
                continue

            real_c = real_images[idx]  # [Nr, 1, H, W]

            gen_labels = torch.full((cfg.gen_per_class_for_eval,), c, dtype=torch.long, device=cfg.device)
            fake_c = generate_by_labels(generator, gen_labels, cfg)  # [Ng, 1, H, W]

            # 攤平成 vector 後計算 pairwise MSE
            real_flat = real_c.view(real_c.size(0), -1)
            fake_flat = fake_c.view(fake_c.size(0), -1)

            nearest_values = []
            chunk = 32
            for i in range(0, fake_flat.size(0), chunk):
                f = fake_flat[i:i + chunk]
                # [chunk, Nr, dim]
                mse = ((f[:, None, :] - real_flat[None, :, :]) ** 2).mean(dim=2)
                nearest = mse.min(dim=1).values
                nearest_values.append(nearest.cpu())

            nearest_values = torch.cat(nearest_values, dim=0)
            rows.append({
                "class_id": c,
                "num_real": len(idx),
                "nearest_mse": nearest_values.mean().item(),
                "nearest_mse_std": nearest_values.std().item(),
            })

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 8. 生成平衡資料集
# ============================================================

@torch.no_grad()
def generate_balanced_dataset(real_images: torch.Tensor, labels: np.ndarray, generator: nn.Module, cfg: CFG, idx_to_label: Dict):
    """
    使用 Generator 補足少數類別，使每個類別數量接近最大類別數。

    輸出：
    1. generated_synthetic_only.npz
       只包含 GAN 生成的補足資料。
    2. mixedwm38_balanced_by_cwgangp.npz
       包含原始資料 + 生成資料。
    """
    out_dir = os.path.join(cfg.output_dir, "balanced_dataset")
    ensure_dir(out_dir)

    labels_np = np.asarray(labels)
    class_count = np.bincount(labels_np, minlength=cfg.num_classes)
    target_count = int(class_count.max())

    generator.eval()

    synthetic_images = []
    synthetic_labels = []

    print("\n[Generate Balanced Dataset]")
    print("Class counts:", class_count.tolist())
    print("Target count per class:", target_count)

    for c in range(cfg.num_classes):
        need = target_count - int(class_count[c])
        if need <= 0:
            continue

        print(f"  class {c:02d}: generate {need} samples")
        remain = need
        while remain > 0:
            b = min(cfg.batch_size, remain)
            c_labels = torch.full((b,), c, dtype=torch.long, device=cfg.device)
            fake = generate_by_labels(generator, c_labels, cfg)
            fake01 = denorm_img(fake).cpu().numpy()  # [B,1,H,W], [0,1]
            synthetic_images.append(fake01)
            synthetic_labels.append(np.full((b,), c, dtype=np.int64))

            # 也可以輸出 png，方便人工檢查
            if cfg.save_generated_images:
                img_dir = os.path.join(out_dir, "generated_png", f"class_{c:02d}")
                ensure_dir(img_dir)
                start_id = need - remain
                for i in range(b):
                    save_image(torch.from_numpy(fake01[i]), os.path.join(img_dir, f"gen_{start_id + i:05d}.png"))

            remain -= b

    if len(synthetic_images) == 0:
        print("所有類別已經平衡，不需要生成補足資料。")
        return

    synthetic_images = np.concatenate(synthetic_images, axis=0)  # [Ns,1,H,W]
    synthetic_labels = np.concatenate(synthetic_labels, axis=0)  # [Ns]

    # 儲存 synthetic only
    np.savez_compressed(
        os.path.join(out_dir, "generated_synthetic_only.npz"),
        images=synthetic_images,
        labels=synthetic_labels,
        idx_to_label=np.array([idx_to_label.get(i, i) for i in range(cfg.num_classes)], dtype=object),
    )

    # 原始資料轉 [0,1]
    real01 = denorm_img(real_images).cpu().numpy()
    real_labels = labels_np.astype(np.int64)

    balanced_images = np.concatenate([real01, synthetic_images], axis=0)
    balanced_labels = np.concatenate([real_labels, synthetic_labels], axis=0)

    np.savez_compressed(
        os.path.join(out_dir, "mixedwm38_balanced_by_cwgangp.npz"),
        images=balanced_images,
        labels=balanced_labels,
        idx_to_label=np.array([idx_to_label.get(i, i) for i in range(cfg.num_classes)], dtype=object),
    )

    print("\n已輸出：")
    print("  generated_synthetic_only.npz")
    print("  mixedwm38_balanced_by_cwgangp.npz")
    print(f"  synthetic shape = {synthetic_images.shape}")
    print(f"  balanced shape  = {balanced_images.shape}")


# ============================================================
# 9. 訓練主流程
# ============================================================

def train(cfg: CFG):
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    print("=" * 80)
    print("Conditional WGAN-GP for MixedWM38")
    print("Device:", cfg.device)
    print("Output:", cfg.output_dir)
    print("=" * 80)

    # ---------- 讀取資料 ----------
    print_npz_keys(cfg.npz_path)
    images_np, labels_np = auto_load_npz(cfg.npz_path)
    labels_int, idx_to_label = convert_labels_to_int(labels_np)

    detected_classes = len(set(labels_int.tolist()))
    if detected_classes != cfg.num_classes:
        print(f"[Warning] 偵測到資料類別數 = {detected_classes}，但 cfg.num_classes = {cfg.num_classes}")
        print("如果你的資料不是 38 類，請修改 CFG.num_classes。")

    images_tensor = preprocess_images(images_np, cfg.image_size)

    print("[Dataset]")
    print("  images tensor:", tuple(images_tensor.shape), images_tensor.dtype)
    print("  labels:", labels_int.shape, labels_int.dtype)
    print("  label mapping:", idx_to_label)
    print("  class counts:", np.bincount(labels_int, minlength=cfg.num_classes).tolist())

    dataset = WaferMapDataset(images_tensor, labels_int)
    loader = build_dataloader(dataset, cfg)

    # ---------- 建立模型 ----------
    G = Generator(cfg.latent_dim, cfg.num_classes, cfg.label_emb_dim, cfg.base_ch).to(cfg.device)
    D = Critic(cfg.num_classes, cfg.image_size, cfg.base_ch).to(cfg.device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    history = {"g_loss": [], "d_loss": [], "gp": [], "wasserstein": []}

    # ---------- 訓練 ----------
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()

        epoch_g_loss = []
        epoch_d_loss = []
        epoch_gp = []
        epoch_wdist = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for real, labels in pbar:
            real = real.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            b = real.size(0)

            # ====================================================
            # Step 1: 訓練 Critic
            # ====================================================
            for _ in range(cfg.n_critic):
                z = torch.randn(b, cfg.latent_dim, device=cfg.device)
                fake = G(z, labels).detach()

                real_score = D(real, labels)
                fake_score = D(fake, labels)

                gp = gradient_penalty(D, real, fake, labels, cfg.device)

                # WGAN-GP critic loss
                # 希望 real_score 大，fake_score 小
                d_loss = fake_score.mean() - real_score.mean() + cfg.lambda_gp * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # ====================================================
            # Step 2: 訓練 Generator
            # ====================================================
            z = torch.randn(b, cfg.latent_dim, device=cfg.device)
            fake = G(z, labels)
            fake_score = D(fake, labels)

            # Generator 希望 fake 被 Critic 給高分
            g_loss = -fake_score.mean()

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            wasserstein_dist = real_score.mean().item() - fake_score.mean().item()

            epoch_g_loss.append(g_loss.item())
            epoch_d_loss.append(d_loss.item())
            epoch_gp.append(gp.item())
            epoch_wdist.append(wasserstein_dist)

            global_step += 1
            pbar.set_postfix({
                "G": f"{g_loss.item():.4f}",
                "D": f"{d_loss.item():.4f}",
                "GP": f"{gp.item():.4f}",
                "W": f"{wasserstein_dist:.4f}",
            })

        history["g_loss"].append(float(np.mean(epoch_g_loss)))
        history["d_loss"].append(float(np.mean(epoch_d_loss)))
        history["gp"].append(float(np.mean(epoch_gp)))
        history["wasserstein"].append(float(np.mean(epoch_wdist)))

        print(
            f"[Epoch {epoch:03d}] "
            f"G={history['g_loss'][-1]:.4f}, "
            f"D={history['d_loss'][-1]:.4f}, "
            f"GP={history['gp'][-1]:.4f}, "
            f"W={history['wasserstein'][-1]:.4f}"
        )

        # ---------- 每隔幾個 epoch 輸出生成圖與 checkpoint ----------
        if epoch % cfg.sample_every == 0 or epoch == 1 or epoch == cfg.epochs:
            save_sample_grid(G, epoch, cfg)
            plot_loss_curve(history, cfg)

            ckpt = {
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "cfg": cfg.__dict__,
                "history": history,
                "idx_to_label": idx_to_label,
            }
            torch.save(ckpt, os.path.join(cfg.output_dir, "last_checkpoint.pth"))

    # ---------- 儲存最終模型 ----------
    torch.save(G.state_dict(), os.path.join(cfg.output_dir, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(cfg.output_dir, "critic_final.pth"))

    # ---------- MSE 評估 ----------
    print("\n[MSE Evaluation]")
    mse_mean_df = compute_class_mean_mse(images_tensor, labels_int, G, cfg)
    mse_nearest_df = compute_nearest_real_mse(images_tensor, labels_int, G, cfg)

    mse_df = mse_mean_df.merge(mse_nearest_df, on=["class_id", "num_real"], how="outer")
    mse_csv_path = os.path.join(cfg.output_dir, "mse_evaluation.csv")
    mse_df.to_csv(mse_csv_path, index=False, encoding="utf-8-sig")
    print(mse_df)
    print("MSE report saved to:", mse_csv_path)

    # ---------- 真實圖 vs 生成圖比較 ----------
    save_real_fake_compare(images_tensor, labels_int, G, cfg)

    # ---------- 輸出平衡後資料集 ----------
    if cfg.save_balanced_npz:
        generate_balanced_dataset(images_tensor, labels_int, G, cfg, idx_to_label)

    print("\n全部完成。")
    print("主要輸出檔案：")
    print("  loss_curve.png")
    print("  samples/epoch_xxxx.png")
    print("  compare_real_fake/class_xx_real_top_fake_bottom.png")
    print("  mse_evaluation.csv")
    print("  generator_final.pth")
    print("  critic_final.pth")


# ============================================================
# 10. 主程式入口
# ============================================================

if __name__ == "__main__":
    cfg = CFG()

    # 你最常需要修改的是這裡：
    cfg.npz_path = r"./datasets/MixedWM38.npz"
    cfg.output_dir = r"./outputs_cwgangp_mixedwm38_v2"

    # 如果你的 GPU 記憶體不足，可以把 batch_size 改小，例如 32 或 16。
    cfg.batch_size = 64

    # 作業先跑 100 epochs 即可；如果生成圖不穩，再改 200~300。
    # cfg.epochs = 100

    cfg.epochs = 60
    cfg.lr_g = 1e-4
    cfg.lr_d = 5e-5
    cfg.n_critic = 3

    train(cfg)
