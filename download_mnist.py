import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ==============================
# 1. 基本設定
# ==============================

# MNIST 下載位置
DATA_DIR = "./datasets"

# 圖片輸出資料夾
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 批次大小
BATCH_SIZE = 64


# ==============================
# 2. 定義資料轉換
# ==============================
# Autoencoder 的目標是重建影像，
# 因此這裡只使用 ToTensor()，將影像轉成 [0, 1] 的 tensor。
#
# 注意：
# 這裡先不要使用 Normalize(mean, std)，
# 因為後面 Autoencoder 的輸出通常會接 sigmoid，
# 輸出範圍也是 [0, 1]，這樣比較方便計算 MSE 或 BCE loss。

transform = transforms.Compose([
    transforms.ToTensor()
])


# ==============================
# 3. 下載 MNIST 資料集
# ==============================

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


# ==============================
# 4. 建立 DataLoader
# ==============================

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


# ==============================
# 5. 檢查資料資訊
# ==============================

print("MNIST dataset downloaded successfully.")
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images : {len(test_dataset)}")

# 取出一個 batch
images, labels = next(iter(train_loader))

print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")
print(f"Image pixel min value: {images.min().item():.4f}")
print(f"Image pixel max value: {images.max().item():.4f}")


# ==============================
# 6. 顯示並儲存部分 MNIST 影像
# ==============================

plt.figure(figsize=(8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)

    # images[i] shape: [1, 28, 28]
    # squeeze() 後變成 [28, 28]，方便 matplotlib 顯示
    img = images[i].squeeze(0)

    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "mnist_samples.png")
plt.savefig(save_path)
plt.show()

print(f"Sample image saved to: {save_path}")