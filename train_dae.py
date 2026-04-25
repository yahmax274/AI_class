import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================================================
# 1. 基本設定
# =========================================================

DATA_DIR = "./datasets"
OUTPUT_DIR = "./outputs/dae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

# 雜訊強度
# 數值越大，輸入影像越模糊、越難還原
# MNIST 可以先用 0.3 或 0.4
NOISE_FACTOR = 0.4

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
# 3. 加入雜訊的函式
# =========================================================

def add_noise(images, noise_factor=0.4):
    """
    對輸入影像加入 Gaussian noise。

    images:
        原始乾淨影像，範圍為 [0, 1]

    noise_factor:
        控制雜訊強度

    return:
        noisy_images，加入雜訊後的影像，範圍仍限制在 [0, 1]
    """

    noise = noise_factor * torch.randn_like(images)

    noisy_images = images + noise

    # 將像素值限制在 [0, 1]
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

    return noisy_images


# =========================================================
# 4. 定義 DAE 模型
# =========================================================
# DAE 的模型架構可以和 CAE 一樣。
# 差別不是模型架構，而是訓練資料：
# 輸入 noisy image，目標是 clean image。

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder, DAE

    Input : noisy image  [B, 1, 28, 28]
    Output: clean image  [B, 1, 28, 28]

    Encoder:
        [B, 1, 28, 28]
        -> [B, 16, 14, 14]
        -> [B, 32, 7, 7]

    Decoder:
        [B, 32, 7, 7]
        -> [B, 16, 14, 14]
        -> [B, 1, 28, 28]
    """

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # -------------------------------
        # Encoder：壓縮 noisy image 的特徵
        # -------------------------------
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 16, 14, 14]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # [B, 16, 14, 14] -> [B, 32, 7, 7]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # -------------------------------
        # Decoder：將特徵還原成乾淨影像
        # -------------------------------
        self.decoder = nn.Sequential(
            # [B, 32, 7, 7] -> [B, 16, 14, 14]
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            # [B, 16, 14, 14] -> [B, 1, 28, 28]
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            # 輸出影像限制在 [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        denoised = self.decoder(latent)
        return denoised


# =========================================================
# 5. 建立模型、Loss、Optimizer
# =========================================================

model = DenoisingAutoencoder().to(DEVICE)

# DAE 的目標是讓 denoised image 接近 clean image
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)


# =========================================================
# 6. 訓練 DAE
# =========================================================

train_losses = []

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        # DAE 不需要 labels，只需要 images
        clean_images = images.to(DEVICE)

        # 對乾淨影像加入雜訊，作為模型輸入
        noisy_images = add_noise(
            clean_images,
            noise_factor=NOISE_FACTOR
        )

        # -------------------------------
        # Forward
        # -------------------------------
        denoised_images = model(noisy_images)

        # 輸出 denoised_images 要接近 clean_images
        loss = criterion(denoised_images, clean_images)

        # -------------------------------
        # Backward
        # -------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}], Train MSE Loss: {avg_loss:.6f}")


# =========================================================
# 7. 測試集 MSE 評估
# =========================================================

model.eval()

test_loss = 0.0
noisy_input_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        clean_images = images.to(DEVICE)

        noisy_images = add_noise(
            clean_images,
            noise_factor=NOISE_FACTOR
        )

        denoised_images = model(noisy_images)

        # DAE 去雜訊後的 MSE
        denoised_loss = criterion(denoised_images, clean_images)

        # 原本 noisy image 和 clean image 的 MSE
        noisy_loss = criterion(noisy_images, clean_images)

        test_loss += denoised_loss.item() * clean_images.size(0)
        noisy_input_loss += noisy_loss.item() * clean_images.size(0)

test_mse = test_loss / len(test_dataset)
noisy_mse = noisy_input_loss / len(test_dataset)

print(f"\nNoisy Input MSE : {noisy_mse:.6f}")
print(f"DAE Test MSE    : {test_mse:.6f}")

if test_mse < noisy_mse:
    print("Result: DAE successfully reduced the noise.")
else:
    print("Result: DAE did not reduce the noise effectively.")


# =========================================================
# 8. 儲存模型
# =========================================================

model_save_path = os.path.join(OUTPUT_DIR, "dae_model.pth")
torch.save(model.state_dict(), model_save_path)

print(f"DAE model saved to: {model_save_path}")


# =========================================================
# 9. 繪製 Loss Curve
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("DAE Training Loss Curve")
plt.grid(True)
plt.tight_layout()

loss_curve_path = os.path.join(OUTPUT_DIR, "dae_loss_curve.png")
plt.savefig(loss_curve_path)
plt.show()

print(f"Loss curve saved to: {loss_curve_path}")


# =========================================================
# 10. 顯示 Original / Noisy / Denoised 比較
# =========================================================

model.eval()

images, labels = next(iter(test_loader))
clean_images = images.to(DEVICE)

with torch.no_grad():
    noisy_images = add_noise(
        clean_images,
        noise_factor=NOISE_FACTOR
    )

    denoised_images = model(noisy_images)

# 移回 CPU，方便 matplotlib 顯示
clean_images = clean_images.cpu()
noisy_images = noisy_images.cpu()
denoised_images = denoised_images.cpu()

num_images = 10

plt.figure(figsize=(12, 5))

for i in range(num_images):
    # -------------------------------
    # 第一排：原始乾淨影像
    # -------------------------------
    plt.subplot(3, num_images, i + 1)
    plt.imshow(clean_images[i].squeeze(0), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # -------------------------------
    # 第二排：加入雜訊後的影像
    # -------------------------------
    plt.subplot(3, num_images, i + 1 + num_images)
    plt.imshow(noisy_images[i].squeeze(0), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # -------------------------------
    # 第三排：DAE 去雜訊後的影像
    # -------------------------------
    plt.subplot(3, num_images, i + 1 + 2 * num_images)
    plt.imshow(denoised_images[i].squeeze(0), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()

denoise_path = os.path.join(OUTPUT_DIR, "dae_denoising_result.png")
plt.savefig(denoise_path)
plt.show()

print(f"Denoising result saved to: {denoise_path}")


# =========================================================
# 11. 額外測試：不同雜訊強度下的去雜訊效果
# =========================================================
# 這張圖可以讓報告更完整。
# 可以觀察當 noise_factor 變大時，DAE 還能不能有效還原。

noise_levels = [0.1, 0.2, 0.4, 0.6]

# 只取一張影像做展示
single_clean = clean_images[0:1].to(DEVICE)

plt.figure(figsize=(12, 6))

for idx, noise_level in enumerate(noise_levels):
    with torch.no_grad():
        single_noisy = add_noise(
            single_clean,
            noise_factor=noise_level
        )

        single_denoised = model(single_noisy)

    single_noisy = single_noisy.cpu()
    single_denoised = single_denoised.cpu()

    # 第一排：不同 noise level 的 noisy image
    plt.subplot(2, len(noise_levels), idx + 1)
    plt.imshow(single_noisy[0].squeeze(0), cmap="gray")
    plt.title(f"Noisy {noise_level}")
    plt.axis("off")

    # 第二排：DAE denoised result
    plt.subplot(2, len(noise_levels), idx + 1 + len(noise_levels))
    plt.imshow(single_denoised[0].squeeze(0), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()

noise_compare_path = os.path.join(OUTPUT_DIR, "dae_noise_level_comparison.png")
plt.savefig(noise_compare_path)
plt.show()

print(f"Noise level comparison saved to: {noise_compare_path}")