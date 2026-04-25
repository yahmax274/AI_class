import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =========================================================
# 1. 基本設定
# =========================================================

DATA_DIR = "./datasets"

CAE_MODEL_PATH = "./outputs/cae/cae_model.pth"
VAE_MODEL_PATH = "./outputs/vae/vae_model.pth"
DAE_MODEL_PATH = "./outputs/dae/dae_model.pth"

SUMMARY_DIR = "./outputs/summary"
os.makedirs(SUMMARY_DIR, exist_ok=True)

BATCH_SIZE = 128

# 注意：
# 這裡的 LATENT_DIM 必須和你訓練 VAE 時的設定一樣。
# 如果你訓練 VAE 時用 LATENT_DIM = 2，這裡就設 2。
# 如果你後來改成 LATENT_DIM = 16，這裡也要改成 16。
LATENT_DIM = 2

# DAE 測試時使用的 noise factor
# 建議和你訓練 DAE 時相同。
NOISE_FACTOR = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# =========================================================
# 2. 載入 MNIST test set
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Test images: {len(test_dataset)}")


# =========================================================
# 3. 定義 CAE 模型
# =========================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder, CAE

    Input : [B, 1, 28, 28]
    Output: [B, 1, 28, 28]
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# =========================================================
# 4. 定義 VAE 模型
# =========================================================

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder, VAE

    Input : [B, 1, 28, 28]
    Output: [B, 1, 28, 28]
    """

    def __init__(self, latent_dim=2):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.flatten_dim = 64 * 7 * 7

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 64, 7, 7)
        reconstructed = self.decoder(h)
        return reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar, z


# =========================================================
# 5. 定義 DAE 模型
# =========================================================

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder, DAE

    Input : noisy image
    Output: denoised image
    """

    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        denoised = self.decoder(latent)
        return denoised


# =========================================================
# 6. 加入 Gaussian noise
# =========================================================

def add_noise(images, noise_factor=0.4):
    """
    對影像加入 Gaussian noise，並將結果限制在 [0, 1]。
    """

    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

    return noisy_images


# =========================================================
# 7. 載入模型
# =========================================================

def load_model(model, model_path, model_name):
    """
    載入模型權重。
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"{model_name} loaded from: {model_path}")

    return model


cae_model = ConvAutoencoder()
vae_model = ConvVAE(latent_dim=LATENT_DIM)
dae_model = DenoisingAutoencoder()

cae_model = load_model(cae_model, CAE_MODEL_PATH, "CAE")
vae_model = load_model(vae_model, VAE_MODEL_PATH, "VAE")
dae_model = load_model(dae_model, DAE_MODEL_PATH, "DAE")


# =========================================================
# 8. 評估 CAE / VAE / DAE 的 MSE
# =========================================================
# 這裡統一使用 nn.MSELoss(reduction="mean")
# 所以 CAE、VAE、DAE 的 MSE 是公平的比較方式。

criterion = nn.MSELoss(reduction="mean")


def evaluate_models():
    """
    使用同一個 test set 評估三個模型。
    """

    cae_total_loss = 0.0
    vae_total_loss = 0.0
    dae_total_loss = 0.0
    noisy_total_loss = 0.0

    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            clean_images = images.to(DEVICE)
            batch_size = clean_images.size(0)

            # -------------------------------
            # CAE reconstruction
            # -------------------------------
            cae_recon = cae_model(clean_images)
            cae_loss = criterion(cae_recon, clean_images)

            # -------------------------------
            # VAE reconstruction
            # -------------------------------
            vae_recon, mu, logvar, z = vae_model(clean_images)
            vae_loss = criterion(vae_recon, clean_images)

            # -------------------------------
            # DAE denoising
            # -------------------------------
            noisy_images = add_noise(
                clean_images,
                noise_factor=NOISE_FACTOR
            )

            dae_denoised = dae_model(noisy_images)
            dae_loss = criterion(dae_denoised, clean_images)

            # noisy input 本身和 clean image 的差距
            noisy_loss = criterion(noisy_images, clean_images)

            # 因為 criterion 是 mean，
            # 所以這裡乘上 batch size，最後再除以 total samples。
            cae_total_loss += cae_loss.item() * batch_size
            vae_total_loss += vae_loss.item() * batch_size
            dae_total_loss += dae_loss.item() * batch_size
            noisy_total_loss += noisy_loss.item() * batch_size

            total_samples += batch_size

    results = {
        "CAE Reconstruction MSE": cae_total_loss / total_samples,
        "VAE Reconstruction MSE": vae_total_loss / total_samples,
        "Noisy Input MSE": noisy_total_loss / total_samples,
        "DAE Denoised MSE": dae_total_loss / total_samples,
    }

    return results


results = evaluate_models()


# =========================================================
# 9. 印出結果
# =========================================================

print("\n========== Model Comparison Results ==========")
for name, value in results.items():
    print(f"{name}: {value:.6f}")


# =========================================================
# 10. 儲存結果成 CSV
# =========================================================

csv_path = os.path.join(SUMMARY_DIR, "model_comparison.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow(["Model / Input", "MSE"])

    for name, value in results.items():
        writer.writerow([name, f"{value:.6f}"])

print(f"\nComparison CSV saved to: {csv_path}")


# =========================================================
# 11. 畫 MSE 比較長條圖
# =========================================================

plt.figure(figsize=(9, 5))

names = list(results.keys())
values = list(results.values())

plt.bar(names, values)
plt.ylabel("MSE")
plt.title("Autoencoder MSE Comparison on MNIST Test Set")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y")
plt.tight_layout()

bar_path = os.path.join(SUMMARY_DIR, "mse_comparison_bar.png")
plt.savefig(bar_path)
plt.show()

print(f"MSE comparison bar chart saved to: {bar_path}")


# =========================================================
# 12. 顯示 Original / CAE / VAE / Noisy / DAE 比較圖
# =========================================================

def save_visual_comparison():
    """
    產生總比較圖：
    Row 1: Original
    Row 2: CAE Reconstruction
    Row 3: VAE Reconstruction
    Row 4: Noisy Input
    Row 5: DAE Denoised
    """

    images, labels = next(iter(test_loader))
    clean_images = images.to(DEVICE)

    with torch.no_grad():
        cae_recon = cae_model(clean_images)

        vae_recon, mu, logvar, z = vae_model(clean_images)

        noisy_images = add_noise(
            clean_images,
            noise_factor=NOISE_FACTOR
        )

        dae_denoised = dae_model(noisy_images)

    clean_images = clean_images.cpu()
    cae_recon = cae_recon.cpu()
    vae_recon = vae_recon.cpu()
    noisy_images = noisy_images.cpu()
    dae_denoised = dae_denoised.cpu()

    num_images = 10

    plt.figure(figsize=(14, 8))

    row_titles = [
        "Original",
        "CAE",
        "VAE",
        "Noisy",
        "DAE"
    ]

    image_rows = [
        clean_images,
        cae_recon,
        vae_recon,
        noisy_images,
        dae_denoised
    ]

    for row_idx, row_images in enumerate(image_rows):
        for col_idx in range(num_images):
            plot_idx = row_idx * num_images + col_idx + 1

            plt.subplot(len(image_rows), num_images, plot_idx)
            plt.imshow(row_images[col_idx].squeeze(0), cmap="gray")
            plt.axis("off")

            if col_idx == 0:
                plt.ylabel(
                    row_titles[row_idx],
                    fontsize=12,
                    rotation=0,
                    labelpad=35,
                    va="center"
                )

    plt.suptitle(
        "Original / CAE / VAE / Noisy / DAE Comparison",
        fontsize=16
    )

    plt.tight_layout()

    compare_path = os.path.join(
        SUMMARY_DIR,
        "autoencoder_visual_comparison.png"
    )

    plt.savefig(compare_path)
    plt.show()

    print(f"Visual comparison saved to: {compare_path}")


save_visual_comparison()


# =========================================================
# 13. 產生簡單文字報告
# =========================================================

report_path = os.path.join(SUMMARY_DIR, "summary_report.txt")

with open(report_path, mode="w", encoding="utf-8") as f:
    f.write("Autoencoder Summary Report\n")
    f.write("==========================\n\n")

    f.write("Dataset: MNIST test set\n")
    f.write(f"Number of test images: {len(test_dataset)}\n")
    f.write(f"DAE noise factor: {NOISE_FACTOR}\n")
    f.write(f"VAE latent dimension: {LATENT_DIM}\n\n")

    f.write("MSE Results:\n")
    for name, value in results.items():
        f.write(f"- {name}: {value:.6f}\n")

    f.write("\nInterpretation:\n")
    f.write(
        "CAE focuses only on image reconstruction, so it usually produces clearer reconstructed images.\n"
    )
    f.write(
        "VAE learns a regularized latent distribution, so its reconstructions may be blurrier, especially when the latent dimension is small.\n"
    )
    f.write(
        "DAE is trained with noisy inputs and clean targets, so its performance should be compared with the noisy input MSE.\n"
    )
    f.write(
        "If the DAE denoised MSE is lower than the noisy input MSE, it means the DAE successfully reduced noise.\n"
    )

print(f"Summary text report saved to: {report_path}")