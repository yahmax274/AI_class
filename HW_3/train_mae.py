import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================================================
# 1. 基本設定
# =========================================================

DATA_DIR = "./datasets"
OUTPUT_DIR = "./outputs/mae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3

# MNIST: 28 x 28
# patch_size = 7 -> 4 x 4 patches
PATCH_SIZE = 7

# 每張圖遮住多少比例的 patch
MASK_RATIO = 0.25

# MAE 主要看 masked region reconstruction
# 但加一點 full image loss 可以讓整張輸出更穩定
FULL_LOSS_WEIGHT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# =========================================================
# 2. 載入 MNIST 資料集
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train images: {len(train_dataset)}")
print(f"Test images : {len(test_dataset)}")


# =========================================================
# 3. 建立 patch mask 函式
# =========================================================

def create_patch_mask(images, patch_size=7, mask_ratio=0.5):
    """
    對 MNIST 影像產生 patch-level mask。

    images:
        原始影像，shape = [B, 1, 28, 28]

    patch_size:
        每個 patch 的大小，例如 7 表示 7x7 patch。

    mask_ratio:
        要遮住的 patch 比例，例如 0.5 表示遮住 50%。

    return:
        masked_images:
            被遮住後的影像。

        mask:
            mask = 1 表示該位置被遮住。
            mask = 0 表示該位置可見。
    """

    B, C, H, W = images.shape

    assert H % patch_size == 0, "Image height must be divisible by patch_size."
    assert W % patch_size == 0, "Image width must be divisible by patch_size."

    grid_h = H // patch_size
    grid_w = W // patch_size

    num_patches = grid_h * grid_w
    num_mask = int(num_patches * mask_ratio)

    # patch_mask shape: [B, num_patches]
    patch_mask = torch.zeros(
        B,
        num_patches,
        device=images.device
    )

    # 對每一張圖隨機選 patch 來遮住
    for i in range(B):
        mask_indices = torch.randperm(
            num_patches,
            device=images.device
        )[:num_mask]

        patch_mask[i, mask_indices] = 1.0

    # [B, num_patches] -> [B, 1, grid_h, grid_w]
    patch_mask = patch_mask.view(B, 1, grid_h, grid_w)

    # 將 patch-level mask 放大回 pixel-level mask
    # [B, 1, 4, 4] -> [B, 1, 28, 28]
    mask = patch_mask.repeat_interleave(
        patch_size,
        dim=2
    ).repeat_interleave(
        patch_size,
        dim=3
    )

    # mask = 1 的地方會被遮住，所以乘上 1 - mask
    masked_images = images * (1.0 - mask)

    return masked_images, mask


# =========================================================
# 4. 定義 MAE 模型
# =========================================================
# 這裡的 MAE 是簡化版 CNN-based Masked Autoencoder。
#
# 輸入不是只有 masked image，而是：
#   channel 1: masked image
#   channel 2: mask map
#
# 這樣模型可以知道哪些地方是真的黑色背景，
# 哪些地方是被刻意遮住的區域。

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder, MAE

    Input:
        [B, 2, 28, 28]
        channel 0 = masked image
        channel 1 = mask map

    Output:
        [B, 1, 28, 28]
        reconstructed image
    """

    def __init__(self):
        super(MaskedAutoencoder, self).__init__()

        # -------------------------------
        # Encoder
        # -------------------------------
        self.encoder = nn.Sequential(
            # [B, 2, 28, 28] -> [B, 32, 14, 14]
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # [B, 32, 14, 14] -> [B, 64, 7, 7]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # bottleneck feature refinement
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # -------------------------------
        # Decoder
        # -------------------------------
        self.decoder = nn.Sequential(
            # [B, 64, 7, 7] -> [B, 32, 14, 14]
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            # [B, 32, 14, 14] -> [B, 1, 28, 28]
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            # MNIST pixel value range: [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# =========================================================
# 5. 定義 MAE loss
# =========================================================

def mae_loss_function(reconstructed, clean_images, mask):
    """
    改良版 MAE loss：
    對白色筆畫區域給更高權重，避免模型只學會預測黑色背景。
    """

    # clean_images 越亮，權重越高
    # 背景大約權重 1，白色筆畫最高權重約 6
    foreground_weight = 1.0 + 5.0 * clean_images

    # masked region weighted MSE
    weighted_mask = mask * foreground_weight

    masked_mse = ((reconstructed - clean_images) ** 2 * weighted_mask).sum()
    masked_mse = masked_mse / weighted_mask.sum().clamp_min(1.0)

    # full image MSE 也加入 foreground weighting
    full_weight = foreground_weight
    full_mse = ((reconstructed - clean_images) ** 2 * full_weight).sum()
    full_mse = full_mse / full_weight.sum().clamp_min(1.0)

    total_loss = masked_mse + FULL_LOSS_WEIGHT * full_mse

    return total_loss, masked_mse, full_mse


# =========================================================
# 6. 建立模型與 optimizer
# =========================================================

model = MaskedAutoencoder().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)


# =========================================================
# 7. 訓練 MAE
# =========================================================

train_total_losses = []
train_masked_losses = []
train_full_losses = []

for epoch in range(EPOCHS):
    model.train()

    total_loss_epoch = 0.0
    masked_loss_epoch = 0.0
    full_loss_epoch = 0.0

    for images, labels in train_loader:
        clean_images = images.to(DEVICE)

        # 建立 masked image 和 mask
        masked_images, mask = create_patch_mask(
            clean_images,
            patch_size=PATCH_SIZE,
            mask_ratio=MASK_RATIO
        )

        # MAE input = masked image + mask map
        mae_input = torch.cat(
            [masked_images, mask],
            dim=1
        )

        # -------------------------------
        # Forward
        # -------------------------------
        reconstructed = model(mae_input)

        loss, masked_loss, full_loss = mae_loss_function(
            reconstructed,
            clean_images,
            mask
        )

        # -------------------------------
        # Backward
        # -------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = clean_images.size(0)

        total_loss_epoch += loss.item() * batch_size
        masked_loss_epoch += masked_loss.item() * batch_size
        full_loss_epoch += full_loss.item() * batch_size

    avg_total_loss = total_loss_epoch / len(train_dataset)
    avg_masked_loss = masked_loss_epoch / len(train_dataset)
    avg_full_loss = full_loss_epoch / len(train_dataset)

    train_total_losses.append(avg_total_loss)
    train_masked_losses.append(avg_masked_loss)
    train_full_losses.append(avg_full_loss)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Total Loss: {avg_total_loss:.6f}, "
        f"Masked MSE: {avg_masked_loss:.6f}, "
        f"Full MSE: {avg_full_loss:.6f}"
    )


# =========================================================
# 8. 測試集評估
# =========================================================

model.eval()

masked_sse = 0.0
masked_count = 0.0

completed_total_mse = 0.0
masked_input_total_mse = 0.0

total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        clean_images = images.to(DEVICE)

        masked_images, mask = create_patch_mask(
            clean_images,
            patch_size=PATCH_SIZE,
            mask_ratio=MASK_RATIO
        )

        mae_input = torch.cat(
            [masked_images, mask],
            dim=1
        )

        reconstructed = model(mae_input)

        # 只把被遮住的地方用模型輸出補回來
        # 可見區域保留原本的 masked_images
        completed_images = masked_images * (1.0 - mask) + reconstructed * mask

        # masked region MSE
        masked_sse += (((reconstructed - clean_images) ** 2) * mask).sum().item()
        masked_count += mask.sum().item()

        # completed image full MSE
        completed_mse = F.mse_loss(
            completed_images,
            clean_images,
            reduction="mean"
        )

        # masked input 和 clean image 的差距
        masked_input_mse = F.mse_loss(
            masked_images,
            clean_images,
            reduction="mean"
        )

        batch_size = clean_images.size(0)

        completed_total_mse += completed_mse.item() * batch_size
        masked_input_total_mse += masked_input_mse.item() * batch_size
        total_samples += batch_size

masked_region_mse = masked_sse / max(masked_count, 1.0)
completed_test_mse = completed_total_mse / total_samples
masked_input_mse = masked_input_total_mse / total_samples

print("\n========== MAE Test Results ==========")
print(f"Masked Input MSE      : {masked_input_mse:.6f}")
print(f"MAE Completed MSE     : {completed_test_mse:.6f}")
print(f"MAE Masked Region MSE : {masked_region_mse:.6f}")

if completed_test_mse < masked_input_mse:
    print("Result: MAE successfully reconstructed masked regions.")
else:
    print("Result: MAE did not improve over the masked input.")


# =========================================================
# 9. 儲存模型
# =========================================================

model_save_path = os.path.join(OUTPUT_DIR, "mae_model.pth")
torch.save(model.state_dict(), model_save_path)

print(f"\nMAE model saved to: {model_save_path}")


# =========================================================
# 10. 繪製 loss curve
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, EPOCHS + 1),
    train_total_losses,
    marker="o",
    label="Total Loss"
)
plt.plot(
    range(1, EPOCHS + 1),
    train_masked_losses,
    marker="o",
    label="Masked MSE"
)
plt.plot(
    range(1, EPOCHS + 1),
    train_full_losses,
    marker="o",
    label="Full MSE"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MAE Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_curve_path = os.path.join(OUTPUT_DIR, "mae_loss_curve.png")
plt.savefig(loss_curve_path)
plt.show()

print(f"Loss curve saved to: {loss_curve_path}")


# =========================================================
# 11. 顯示 Original / Mask / Masked / Reconstructed / Completed
# =========================================================

model.eval()

# 固定 visual seed，讓每次展示比較穩定
torch.manual_seed(42)

images, labels = next(iter(test_loader))
clean_images = images.to(DEVICE)

with torch.no_grad():
    masked_images, mask = create_patch_mask(
        clean_images,
        patch_size=PATCH_SIZE,
        mask_ratio=MASK_RATIO
    )

    mae_input = torch.cat(
        [masked_images, mask],
        dim=1
    )

    reconstructed = model(mae_input)

    completed_images = masked_images * (1.0 - mask) + reconstructed * mask

clean_images = clean_images.cpu()
masked_images = masked_images.cpu()
mask = mask.cpu()
reconstructed = reconstructed.cpu()
completed_images = completed_images.cpu()

num_images = 10

plt.figure(figsize=(14, 8))

rows = [
    ("Original", clean_images),
    ("Mask", mask),
    ("Masked", masked_images),
    ("MAE Output", reconstructed),
    ("Completed", completed_images),
]

for row_idx, (row_name, row_images) in enumerate(rows):
    for col_idx in range(num_images):
        plot_idx = row_idx * num_images + col_idx + 1

        plt.subplot(len(rows), num_images, plot_idx)
        plt.imshow(row_images[col_idx].squeeze(0), cmap="gray")
        plt.axis("off")

        if col_idx == 0:
            plt.ylabel(
                row_name,
                fontsize=12,
                rotation=0,
                labelpad=35,
                va="center"
            )

plt.suptitle("MAE Reconstruction Results", fontsize=16)
plt.tight_layout()

recon_path = os.path.join(OUTPUT_DIR, "mae_reconstruction_result.png")
plt.savefig(recon_path)
plt.show()

print(f"MAE reconstruction result saved to: {recon_path}")


# =========================================================
# 12. 額外測試：不同 mask ratio 的重建效果
# =========================================================

mask_ratios = [0.25, 0.5, 0.75]

single_clean = clean_images[0:1].to(DEVICE)

plt.figure(figsize=(12, 6))

for idx, ratio in enumerate(mask_ratios):
    with torch.no_grad():
        single_masked, single_mask = create_patch_mask(
            single_clean,
            patch_size=PATCH_SIZE,
            mask_ratio=ratio
        )

        single_input = torch.cat(
            [single_masked, single_mask],
            dim=1
        )

        single_reconstructed = model(single_input)

        single_completed = (
            single_masked * (1.0 - single_mask)
            + single_reconstructed * single_mask
        )

    single_masked = single_masked.cpu()
    single_completed = single_completed.cpu()

    # 第一排：masked image
    plt.subplot(2, len(mask_ratios), idx + 1)
    plt.imshow(single_masked[0].squeeze(0), cmap="gray")
    plt.title(f"Masked {ratio}")
    plt.axis("off")

    # 第二排：completed image
    plt.subplot(2, len(mask_ratios), idx + 1 + len(mask_ratios))
    plt.imshow(single_completed[0].squeeze(0), cmap="gray")
    plt.title("Completed")
    plt.axis("off")

plt.tight_layout()

ratio_path = os.path.join(OUTPUT_DIR, "mae_mask_ratio_comparison.png")
plt.savefig(ratio_path)
plt.show()

print(f"Mask ratio comparison saved to: {ratio_path}")