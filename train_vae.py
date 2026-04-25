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
OUTPUT_DIR = "./outputs/vae"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3

# latent_dim = 2 的好處是可以直接畫 2D latent space
# 缺點是重建影像可能比 CAE 模糊，這是正常現象
LATENT_DIM = 2

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
# 3. 定義 VAE 模型
# =========================================================

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder, VAE

    Input : [B, 1, 28, 28]
    Output: [B, 1, 28, 28]

    Encoder:
        [B, 1, 28, 28]
        -> [B, 32, 14, 14]
        -> [B, 64, 7, 7]
        -> flatten
        -> mu, logvar

    Latent:
        z = mu + std * epsilon

    Decoder:
        z
        -> [B, 64, 7, 7]
        -> [B, 32, 14, 14]
        -> [B, 1, 28, 28]
    """

    def __init__(self, latent_dim=2):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        # -------------------------------
        # Encoder：將影像壓縮成特徵
        # -------------------------------
        self.encoder = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 32, 14, 14]
            nn.Conv2d(
                in_channels=1,
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
            nn.ReLU(inplace=True)
        )

        # encoder 最後的 feature map 大小是 [64, 7, 7]
        self.flatten_dim = 64 * 7 * 7

        # VAE 不直接輸出 latent vector
        # 而是輸出 latent distribution 的 mu 和 logvar
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # -------------------------------
        # Decoder：從 latent vector 還原影像
        # -------------------------------
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

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

            # 將輸出限制在 [0, 1]
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        輸入影像後，輸出 mu 和 logvar
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick

        原本：
            z ~ N(mu, sigma)

        改寫成：
            z = mu + sigma * epsilon
            epsilon ~ N(0, 1)

        這樣可以讓 sampling 過程仍然可以反向傳播。
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + std * epsilon
        return z

    def decode(self, z):
        """
        將 latent vector z 還原成影像
        """
        h = self.decoder_input(z)
        h = h.view(h.size(0), 64, 7, 7)

        reconstructed = self.decoder(h)
        return reconstructed

    def forward(self, x):
        """
        VAE forward 流程：
        1. encode x 得到 mu, logvar
        2. 使用 reparameterization trick 取樣 z
        3. decode z 得到 reconstructed image
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar, z


# =========================================================
# 4. 定義 VAE Loss
# =========================================================

def vae_loss_function(reconstructed, images, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence

    Reconstruction Loss:
        使用 Binary Cross Entropy，讓 reconstructed 接近原圖。

    KL Divergence:
        讓 latent distribution 接近標準常態分布 N(0, 1)。
    """

    # BCE 使用 sum，這是 VAE 常見寫法
    bce_loss = F.binary_cross_entropy(
        reconstructed,
        images,
        reduction="sum"
    )

    # KL Divergence
    # 公式：
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kld_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = bce_loss + kld_loss

    return total_loss, bce_loss, kld_loss


# =========================================================
# 5. 建立模型、Optimizer
# =========================================================

model = ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

mse_criterion = nn.MSELoss(reduction="sum")

print(model)


# =========================================================
# 6. 訓練 VAE
# =========================================================

train_total_losses = []
train_bce_losses = []
train_kld_losses = []

for epoch in range(EPOCHS):
    model.train()

    total_loss_epoch = 0.0
    total_bce_epoch = 0.0
    total_kld_epoch = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)

        # -------------------------------
        # Forward
        # -------------------------------
        reconstructed, mu, logvar, z = model(images)

        loss, bce_loss, kld_loss = vae_loss_function(
            reconstructed,
            images,
            mu,
            logvar
        )

        # -------------------------------
        # Backward
        # -------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_bce_epoch += bce_loss.item()
        total_kld_epoch += kld_loss.item()

    # 這裡除以資料數量，得到每張影像的平均 loss
    avg_total_loss = total_loss_epoch / len(train_dataset)
    avg_bce_loss = total_bce_epoch / len(train_dataset)
    avg_kld_loss = total_kld_epoch / len(train_dataset)

    train_total_losses.append(avg_total_loss)
    train_bce_losses.append(avg_bce_loss)
    train_kld_losses.append(avg_kld_loss)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Total Loss: {avg_total_loss:.4f}, "
        f"BCE: {avg_bce_loss:.4f}, "
        f"KLD: {avg_kld_loss:.4f}"
    )


# =========================================================
# 7. 測試集 MSE 評估
# =========================================================

model.eval()

test_mse_sum = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        reconstructed, mu, logvar, z = model(images)

        mse_loss = mse_criterion(reconstructed, images)
        test_mse_sum += mse_loss.item()

# 每張影像平均 MSE
test_mse = test_mse_sum / len(test_dataset)

print(f"\nFinal Test MSE: {test_mse:.6f}")


# =========================================================
# 8. 儲存模型
# =========================================================

model_save_path = os.path.join(OUTPUT_DIR, "vae_model.pth")
torch.save(model.state_dict(), model_save_path)

print(f"VAE model saved to: {model_save_path}")


# =========================================================
# 9. 繪製 VAE Loss Curve
# =========================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_total_losses, marker="o", label="Total Loss")
plt.plot(range(1, EPOCHS + 1), train_bce_losses, marker="o", label="BCE Loss")
plt.plot(range(1, EPOCHS + 1), train_kld_losses, marker="o", label="KLD Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

loss_curve_path = os.path.join(OUTPUT_DIR, "vae_loss_curve.png")
plt.savefig(loss_curve_path)
plt.show()

print(f"Loss curve saved to: {loss_curve_path}")


# =========================================================
# 10. 顯示原始影像與重建影像比較
# =========================================================

model.eval()

images, labels = next(iter(test_loader))
images = images.to(DEVICE)

with torch.no_grad():
    reconstructed, mu, logvar, z = model(images)

images = images.cpu()
reconstructed = reconstructed.cpu()

num_images = 10

plt.figure(figsize=(12, 4))

for i in range(num_images):
    # 第一排：原始影像
    plt.subplot(2, num_images, i + 1)
    plt.imshow(images[i].squeeze(0), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # 第二排：VAE 重建影像
    plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(reconstructed[i].squeeze(0), cmap="gray")
    plt.title("VAE")
    plt.axis("off")

plt.tight_layout()

recon_path = os.path.join(OUTPUT_DIR, "vae_reconstruction.png")
plt.savefig(recon_path)
plt.show()

print(f"Reconstruction result saved to: {recon_path}")


# =========================================================
# 11. 視覺化 VAE Latent Space
# =========================================================
# 因為我們設定 LATENT_DIM = 2，所以可以直接畫 2D scatter plot。
# 每一個點代表一張 MNIST 影像在 latent space 中的位置。
# 顏色代表真實數字 label。

model.eval()

all_mu = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        mu, logvar = model.encode(images)

        all_mu.append(mu.cpu())
        all_labels.append(labels)

all_mu = torch.cat(all_mu, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    all_mu[:, 0],
    all_mu[:, 1],
    c=all_labels,
    cmap="tab10",
    s=5,
    alpha=0.7
)

plt.colorbar(scatter, ticks=range(10))
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("VAE Latent Space Visualization")
plt.grid(True)
plt.tight_layout()

latent_path = os.path.join(OUTPUT_DIR, "vae_latent_space.png")
plt.savefig(latent_path)
plt.show()

print(f"Latent space visualization saved to: {latent_path}")


# =========================================================
# 12. 從 Latent Space 產生新影像
# =========================================================
# 因為 latent space 是 2D，所以我們可以在 [-3, 3] 的範圍中取樣，
# 並將每個 z 丟進 decoder，產生新的數字影像。

model.eval()

grid_size = 20
z_range = torch.linspace(-3, 3, grid_size)

generated_images = []

with torch.no_grad():
    for y in z_range:
        row_images = []

        for x in z_range:
            z = torch.tensor([[x, y]], dtype=torch.float32).to(DEVICE)
            generated = model.decode(z)

            row_images.append(generated.cpu().squeeze(0).squeeze(0))

        generated_images.append(row_images)

plt.figure(figsize=(10, 10))

for i in range(grid_size):
    for j in range(grid_size):
        plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
        plt.imshow(generated_images[i][j], cmap="gray")
        plt.axis("off")

plt.suptitle("VAE Generated Samples from 2D Latent Space", fontsize=16)
plt.tight_layout()

generated_path = os.path.join(OUTPUT_DIR, "vae_generated_samples.png")
plt.savefig(generated_path)
plt.show()

print(f"Generated samples saved to: {generated_path}")