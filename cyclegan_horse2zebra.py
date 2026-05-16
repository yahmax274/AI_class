import os
import random
import itertools
import csv
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 1. 基本設定
# ============================================================

@dataclass
class Config:
    # 資料集路徑
    data_root: str = "./datasets/horse2zebra"

    # 輸出資料夾
    output_dir: str = "./cyclegan_outputs"
    sample_dir: str = "./cyclegan_outputs/samples"
    checkpoint_dir: str = "./cyclegan_outputs/checkpoints"

    # 訓練設定
    image_size: int = 256
    batch_size: int = 1
    num_epochs: int = 100
    num_workers: int = 2

    # CycleGAN 常用超參數
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # loss 權重
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0

    # 影像儲存與模型儲存頻率
    sample_interval: int = 5
    save_interval: int = 10

    # Generator residual blocks 數量
    # 256x256 通常使用 9 個 residual blocks
    n_residual_blocks: int = 9

    # 是否使用 GPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 隨機種子
    seed: int = 42


cfg = Config()


# ============================================================
# 2. 建立資料夾與設定 random seed
# ============================================================

def setup_environment(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    print(f"Using device: {cfg.device}")


# ============================================================
# 3. Dataset：非成對 Horse2Zebra 資料讀取
# ============================================================

class UnpairedImageDataset(Dataset):
    """
    CycleGAN 不需要 paired data。
    因此每次會讀取一張 A domain 圖片與一張 B domain 圖片，
    但兩者不需要是一一對應。
    """

    def __init__(self, root, mode="train", transform=None):
        super().__init__()

        self.root = root
        self.mode = mode
        self.transform = transform

        self.dir_A = os.path.join(root, f"{mode}A")
        self.dir_B = os.path.join(root, f"{mode}B")

        self.files_A = self._get_image_files(self.dir_A)
        self.files_B = self._get_image_files(self.dir_B)

        if len(self.files_A) == 0:
            raise RuntimeError(f"找不到 A domain 圖片：{self.dir_A}")

        if len(self.files_B) == 0:
            raise RuntimeError(f"找不到 B domain 圖片：{self.dir_B}")

        print(f"[{mode}] A images: {len(self.files_A)}")
        print(f"[{mode}] B images: {len(self.files_B)}")

    def _get_image_files(self, folder):
        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        files = []

        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_exts:
                files.append(path)

        files.sort()
        return files

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        # B domain 隨機抽取，符合 unpaired training
        path_B = random.choice(self.files_B)

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {
            "A": img_A,
            "B": img_B,
            "path_A": path_A,
            "path_B": path_B
        }


def get_transforms(cfg, mode="train"):
    """
    CycleGAN 常見前處理：
    1. Resize
    2. RandomHorizontalFlip
    3. ToTensor
    4. Normalize 到 [-1, 1]
    """

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])


# ============================================================
# 4. Generator：ResNet-based Generator
# ============================================================

class ResidualBlock(nn.Module):
    """
    CycleGAN Generator 中常用的 Residual Block。
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """
    CycleGAN Generator：
    Encoder → Residual Blocks → Decoder

    輸入：RGB image
    輸出：轉換後的 RGB image
    """

    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super().__init__()

        model = []

        # 初始卷積
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_channels = 64
        out_channels = in_channels * 2

        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels * 2

        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling
        out_channels = in_channels // 2

        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        # 輸出層
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ============================================================
# 5. Discriminator：PatchGAN Discriminator
# ============================================================

class Discriminator(nn.Module):
    """
    CycleGAN 使用 PatchGAN Discriminator。
    它不是只輸出一個真偽分數，而是輸出一張 patch-level 的真假圖。
    """

    def __init__(self, input_channels=3):
        super().__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ]

            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 6. 權重初始化
# ============================================================

def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)

        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================
# 7. Image Pool：穩定 Discriminator 訓練
# ============================================================

class ImagePool:
    """
    CycleGAN 常用技巧：
    Discriminator 不只看最新生成的 fake image，
    也會混合過去生成的 fake image，讓訓練更穩定。
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        output_images = []

        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image)
                output_images.append(image)
            else:
                if random.random() > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    old_image = self.images[random_id].clone()
                    self.images[random_id] = image
                    output_images.append(old_image)
                else:
                    output_images.append(image)

        return torch.cat(output_images, dim=0)


# ============================================================
# 8. 學習率排程器
# ============================================================

class LambdaLR:
    """
    前半段維持固定 learning rate，
    後半段逐漸 decay。
    """

    def __init__(self, num_epochs, offset, decay_start_epoch):
        assert (num_epochs - decay_start_epoch) > 0
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.num_epochs - self.decay_start_epoch
        )


# ============================================================
# 9. 圖片反正規化
# ============================================================

def denormalize(tensor):
    """
    將 [-1, 1] 轉回 [0, 1]，方便存圖。
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# ============================================================
# 10. 儲存訓練 sample 圖
# ============================================================

@torch.no_grad()
def save_sample_images(
    epoch,
    G_A2B,
    G_B2A,
    dataloader,
    cfg,
    max_samples=4
):
    G_A2B.eval()
    G_B2A.eval()

    batch = next(iter(dataloader))

    real_A = batch["A"].to(cfg.device)
    real_B = batch["B"].to(cfg.device)

    real_A = real_A[:max_samples]
    real_B = real_B[:max_samples]

    fake_B = G_A2B(real_A)
    rec_A = G_B2A(fake_B)

    fake_A = G_B2A(real_B)
    rec_B = G_A2B(fake_A)

    # 排列順序：
    # real horse, fake zebra, reconstructed horse
    # real zebra, fake horse, reconstructed zebra
    images = torch.cat([
        denormalize(real_A),
        denormalize(fake_B),
        denormalize(rec_A),
        denormalize(real_B),
        denormalize(fake_A),
        denormalize(rec_B)
    ], dim=0)

    grid = make_grid(images, nrow=max_samples)

    save_path = os.path.join(cfg.sample_dir, f"epoch_{epoch:03d}.png")
    save_image(grid, save_path)

    G_A2B.train()
    G_B2A.train()


# ============================================================
# 11. 儲存 loss curve
# ============================================================

def save_loss_curve(history, cfg):
    csv_path = os.path.join(cfg.output_dir, "loss_history.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "loss_G",
            "loss_GAN",
            "loss_cycle",
            "loss_identity",
            "loss_D_A",
            "loss_D_B"
        ])

        for row in history:
            writer.writerow([
                row["epoch"],
                row["loss_G"],
                row["loss_GAN"],
                row["loss_cycle"],
                row["loss_identity"],
                row["loss_D_A"],
                row["loss_D_B"]
            ])

    epochs = [x["epoch"] for x in history]
    loss_G = [x["loss_G"] for x in history]
    loss_D_A = [x["loss_D_A"] for x in history]
    loss_D_B = [x["loss_D_B"] for x in history]
    loss_cycle = [x["loss_cycle"] for x in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_G, label="Generator Loss")
    plt.plot(epochs, loss_D_A, label="Discriminator A Loss")
    plt.plot(epochs, loss_D_B, label="Discriminator B Loss")
    plt.plot(epochs, loss_cycle, label="Cycle Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CycleGAN Training Loss Curves")
    plt.legend()
    plt.grid(True)

    fig_path = os.path.join(cfg.output_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()


# ============================================================
# 12. 儲存模型
# ============================================================

def save_checkpoint(epoch, G_A2B, G_B2A, D_A, D_B, cfg):
    save_path = os.path.join(cfg.checkpoint_dir, f"cyclegan_epoch_{epoch:03d}.pth")

    torch.save({
        "epoch": epoch,
        "G_A2B": G_A2B.state_dict(),
        "G_B2A": G_B2A.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
    }, save_path)

    print(f"Checkpoint saved: {save_path}")


# ============================================================
# 13. 訓練主程式
# ============================================================

def train(cfg):
    setup_environment(cfg)

    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    train_transform = get_transforms(cfg, mode="train")
    test_transform = get_transforms(cfg, mode="test")

    train_dataset = UnpairedImageDataset(
        root=cfg.data_root,
        mode="train",
        transform=train_transform
    )

    test_dataset = UnpairedImageDataset(
        root=cfg.data_root,
        mode="test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # -----------------------------
    # Models
    # -----------------------------
    G_A2B = GeneratorResNet(
        input_channels=3,
        output_channels=3,
        n_residual_blocks=cfg.n_residual_blocks
    ).to(cfg.device)

    G_B2A = GeneratorResNet(
        input_channels=3,
        output_channels=3,
        n_residual_blocks=cfg.n_residual_blocks
    ).to(cfg.device)

    D_A = Discriminator(input_channels=3).to(cfg.device)
    D_B = Discriminator(input_channels=3).to(cfg.device)

    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # -----------------------------
    # Loss functions
    # -----------------------------
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # -----------------------------
    # Optimizers
    # -----------------------------
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2)
    )

    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2)
    )

    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2)
    )

    # -----------------------------
    # Learning rate schedulers
    # -----------------------------
    decay_start_epoch = cfg.num_epochs // 2

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G,
        lr_lambda=LambdaLR(cfg.num_epochs, 0, decay_start_epoch).step
    )

    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A,
        lr_lambda=LambdaLR(cfg.num_epochs, 0, decay_start_epoch).step
    )

    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B,
        lr_lambda=LambdaLR(cfg.num_epochs, 0, decay_start_epoch).step
    )

    # -----------------------------
    # Image pools
    # -----------------------------
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    history = []

    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(1, cfg.num_epochs + 1):
        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()

        epoch_loss_G = 0.0
        epoch_loss_GAN = 0.0
        epoch_loss_cycle = 0.0
        epoch_loss_identity = 0.0
        epoch_loss_D_A = 0.0
        epoch_loss_D_B = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.num_epochs}]")

        for batch in pbar:
            real_A = batch["A"].to(cfg.device)
            real_B = batch["B"].to(cfg.device)

            # Discriminator 輸出的 patch label
            valid_A = torch.ones_like(D_A(real_A), device=cfg.device)
            fake_label_A = torch.zeros_like(D_A(real_A), device=cfg.device)

            valid_B = torch.ones_like(D_B(real_B), device=cfg.device)
            fake_label_B = torch.zeros_like(D_B(real_B), device=cfg.device)

            # ====================================================
            # 1. Train Generators
            # ====================================================
            optimizer_G.zero_grad()

            # -------------------------
            # Identity Loss
            # -------------------------
            # G_B2A(real_A) 應該仍然像 A
            same_A = G_B2A(real_A)

            # G_A2B(real_B) 應該仍然像 B
            same_B = G_A2B(real_B)

            loss_identity_A = criterion_identity(same_A, real_A)
            loss_identity_B = criterion_identity(same_B, real_B)

            loss_identity = (
                loss_identity_A + loss_identity_B
            ) * cfg.lambda_identity

            # -------------------------
            # GAN Loss
            # -------------------------
            fake_B = G_A2B(real_A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

            fake_A = G_B2A(real_B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            loss_GAN = loss_GAN_A2B + loss_GAN_B2A

            # -------------------------
            # Cycle Consistency Loss
            # -------------------------
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)

            loss_cycle_A = criterion_cycle(rec_A, real_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)

            loss_cycle = (
                loss_cycle_A + loss_cycle_B
            ) * cfg.lambda_cycle

            # -------------------------
            # Total Generator Loss
            # -------------------------
            loss_G = loss_GAN + loss_cycle + loss_identity

            loss_G.backward()
            optimizer_G.step()

            # ====================================================
            # 2. Train Discriminator A
            #    D_A 用來判斷 horse domain 的真假
            # ====================================================
            optimizer_D_A.zero_grad()

            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(
                pred_real_A,
                torch.ones_like(pred_real_A)
            )

            fake_A_for_D = fake_A_pool.query(fake_A.detach())
            pred_fake_A = D_A(fake_A_for_D)
            loss_D_fake_A = criterion_GAN(
                pred_fake_A,
                torch.zeros_like(pred_fake_A)
            )

            loss_D_A = 0.5 * (loss_D_real_A + loss_D_fake_A)

            loss_D_A.backward()
            optimizer_D_A.step()

            # ====================================================
            # 3. Train Discriminator B
            #    D_B 用來判斷 zebra domain 的真假
            # ====================================================
            optimizer_D_B.zero_grad()

            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(
                pred_real_B,
                torch.ones_like(pred_real_B)
            )

            fake_B_for_D = fake_B_pool.query(fake_B.detach())
            pred_fake_B = D_B(fake_B_for_D)
            loss_D_fake_B = criterion_GAN(
                pred_fake_B,
                torch.zeros_like(pred_fake_B)
            )

            loss_D_B = 0.5 * (loss_D_real_B + loss_D_fake_B)

            loss_D_B.backward()
            optimizer_D_B.step()

            # -----------------------------
            # 累積 loss
            # -----------------------------
            epoch_loss_G += loss_G.item()
            epoch_loss_GAN += loss_GAN.item()
            epoch_loss_cycle += loss_cycle.item()
            epoch_loss_identity += loss_identity.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()

            pbar.set_postfix({
                "G": f"{loss_G.item():.4f}",
                "D_A": f"{loss_D_A.item():.4f}",
                "D_B": f"{loss_D_B.item():.4f}",
                "cycle": f"{loss_cycle.item():.4f}"
            })

        # -----------------------------
        # Epoch 平均 loss
        # -----------------------------
        n_batches = len(train_loader)

        epoch_record = {
            "epoch": epoch,
            "loss_G": epoch_loss_G / n_batches,
            "loss_GAN": epoch_loss_GAN / n_batches,
            "loss_cycle": epoch_loss_cycle / n_batches,
            "loss_identity": epoch_loss_identity / n_batches,
            "loss_D_A": epoch_loss_D_A / n_batches,
            "loss_D_B": epoch_loss_D_B / n_batches
        }

        history.append(epoch_record)

        print(
            f"\nEpoch {epoch:03d} | "
            f"G: {epoch_record['loss_G']:.4f} | "
            f"GAN: {epoch_record['loss_GAN']:.4f} | "
            f"Cycle: {epoch_record['loss_cycle']:.4f} | "
            f"Identity: {epoch_record['loss_identity']:.4f} | "
            f"D_A: {epoch_record['loss_D_A']:.4f} | "
            f"D_B: {epoch_record['loss_D_B']:.4f}"
        )

        # -----------------------------
        # 更新 learning rate
        # -----------------------------
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # -----------------------------
        # 儲存 sample image
        # -----------------------------
        if epoch == 1 or epoch % cfg.sample_interval == 0:
            save_sample_images(epoch, G_A2B, G_B2A, test_loader, cfg)

        # -----------------------------
        # 儲存 checkpoint
        # -----------------------------
        if epoch % cfg.save_interval == 0:
            save_checkpoint(epoch, G_A2B, G_B2A, D_A, D_B, cfg)

        # -----------------------------
        # 每個 epoch 都更新 loss curve
        # -----------------------------
        save_loss_curve(history, cfg)

    # 最後儲存一次模型
    save_checkpoint(cfg.num_epochs, G_A2B, G_B2A, D_A, D_B, cfg)

    print("Training finished.")


# ============================================================
# 14. 測試：使用訓練好的 Generator 產生圖片
# ============================================================

@torch.no_grad()
def test(cfg, checkpoint_path):
    setup_environment(cfg)

    test_transform = get_transforms(cfg, mode="test")

    test_dataset = UnpairedImageDataset(
        root=cfg.data_root,
        mode="test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    G_A2B = GeneratorResNet(
        input_channels=3,
        output_channels=3,
        n_residual_blocks=cfg.n_residual_blocks
    ).to(cfg.device)

    G_B2A = GeneratorResNet(
        input_channels=3,
        output_channels=3,
        n_residual_blocks=cfg.n_residual_blocks
    ).to(cfg.device)

    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    G_A2B.load_state_dict(checkpoint["G_A2B"])
    G_B2A.load_state_dict(checkpoint["G_B2A"])

    G_A2B.eval()
    G_B2A.eval()

    test_output_dir = os.path.join(cfg.output_dir, "test_results")
    os.makedirs(test_output_dir, exist_ok=True)

    for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        real_A = batch["A"].to(cfg.device)
        real_B = batch["B"].to(cfg.device)

        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        # A: horse -> zebra
        save_image(
            denormalize(real_A),
            os.path.join(test_output_dir, f"{idx:04d}_real_horse.png")
        )

        save_image(
            denormalize(fake_B),
            os.path.join(test_output_dir, f"{idx:04d}_fake_zebra.png")
        )

        # B: zebra -> horse
        save_image(
            denormalize(real_B),
            os.path.join(test_output_dir, f"{idx:04d}_real_zebra.png")
        )

        save_image(
            denormalize(fake_A),
            os.path.join(test_output_dir, f"{idx:04d}_fake_horse.png")
        )

    print(f"Test results saved to: {test_output_dir}")


# ============================================================
# 15. 主程式入口
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 模式選擇：
    # "train"：訓練 CycleGAN
    # "test" ：使用訓練好的模型產生測試圖片
    # --------------------------------------------------------
    mode = "train"

    if mode == "train":
        train(cfg)

    elif mode == "test":
        # 請改成你訓練好的 checkpoint 路徑
        checkpoint_path = "./cyclegan_outputs/checkpoints/cyclegan_epoch_100.pth"
        test(cfg, checkpoint_path)

    else:
        raise ValueError("mode 必須是 'train' 或 'test'")