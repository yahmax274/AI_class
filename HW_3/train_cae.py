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
OUTPUT_DIR = "./outputs/cae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# =========================================================
# 2. 載入 MNIST 資料集
# =========================================================
# Autoencoder 是重建任務，所以影像輸入與目標輸出都是同一張影像。
# 這裡只使用 ToTensor()，讓 pixel value 保持在 [0, 1]。

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
# 3. 定義 CAE 模型
# =========================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder, CAE

    Input : [B, 1, 28, 28]
    Output: [B, 1, 28, 28]

    Encoder:
        28x28 -> 14x14 -> 7x7

    Decoder:
        7x7 -> 14x14 -> 28x28
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # -------------------------------
        # Encoder：負責壓縮影像特徵
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
        # Decoder：負責還原影像
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

            # 將輸出限制在 [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# =========================================================
# 4. 建立模型、Loss、Optimizer
# =========================================================

model = ConvAutoencoder().to(DEVICE)

# MSELoss 用來計算原圖與重建圖之間的像素差異
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print(model)


# =========================================================
# 5. 訓練 CAE
# =========================================================

train_losses = []

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        # Autoencoder 不需要 label，只需要影像
        images = images.to(DEVICE)

        # -------------------------------
        # Forward
        # -------------------------------
        reconstructed = model(images)

        # CAE 的目標是讓 reconstructed 接近 images
        loss = criterion(reconstructed, images)

        # -------------------------------
        # Backward
        # -------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}], Train MSE Loss: {avg_loss:.6f}")


# =========================================================
# 6. 測試集 MSE 評估
# =========================================================

model.eval()
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        reconstructed = model(images)
        loss = criterion(reconstructed, images)

        test_loss += loss.item() * images.size(0)

test_mse = test_loss / len(test_dataset)

print(f"\nFinal Test MSE: {test_mse:.6f}")


# =========================================================
# 7. 儲存模型
# =========================================================

model_save_path = os.path.join(OUTPUT_DIR, "cae_model.pth")
torch.save(model.state_dict(), model_save_path)

print(f"CAE model saved to: {model_save_path}")


# =========================================================
# 8. 繪製 Loss Curve
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CAE Training Loss Curve")
plt.grid(True)
plt.tight_layout()

loss_curve_path = os.path.join(OUTPUT_DIR, "cae_loss_curve.png")
plt.savefig(loss_curve_path)
plt.show()

print(f"Loss curve saved to: {loss_curve_path}")


# =========================================================
# 9. 顯示原始影像與重建影像比較
# =========================================================

model.eval()

# 取一個 test batch
images, labels = next(iter(test_loader))
images = images.to(DEVICE)

with torch.no_grad():
    reconstructed = model(images)

# 移回 CPU，方便 matplotlib 顯示
images = images.cpu()
reconstructed = reconstructed.cpu()

num_images = 10

plt.figure(figsize=(12, 4))

for i in range(num_images):
    # -------------------------------
    # 第一排：原始影像
    # -------------------------------
    plt.subplot(2, num_images, i + 1)
    plt.imshow(images[i].squeeze(0), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # -------------------------------
    # 第二排：重建影像
    # -------------------------------
    plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(reconstructed[i].squeeze(0), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()

recon_path = os.path.join(OUTPUT_DIR, "cae_reconstruction.png")
plt.savefig(recon_path)
plt.show()

print(f"Reconstruction result saved to: {recon_path}")