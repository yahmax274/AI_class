import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0. 可調整參數
# =========================================================
FILE_PATH = "synthetic_control.data"   # 資料檔路徑
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 80
VAL_RATIO = 0.2            # 從原本 60% training data 中再切出 validation 的比例
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2


# =========================================================
# 1. 固定亂數種子
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. 建立輸出資料夾
# =========================================================
def create_output_dir():
    base_dir = "OUTPUT"
    os.makedirs(base_dir, exist_ok=True)

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


# =========================================================
# 3. 載入資料
# =========================================================
def load_data(file_path):
    """
    synthetic_control.data:
    - 共 600 筆
    - 每筆長度 60
    - 共 6 類，每類 100 筆
    """
    X = np.loadtxt(file_path)   # shape = (600, 60)
    y = np.repeat(np.arange(6), 100)
    return X, y


# =========================================================
# 4. 切分資料
#    先依作業要求每類切:
#    - 60 筆 -> train_val
#    - 40 筆 -> test
#    再從 train_val 中依比例切出 val
# =========================================================
def split_data_by_class(X, y, val_ratio=0.2, seed=42):
    np.random.seed(seed)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for cls in range(6):
        start = cls * 100
        end = (cls + 1) * 100

        X_cls = X[start:end]
        y_cls = y[start:end]

        # 保留原本作業規則：每類 60% / 40%
        X_train_val = X_cls[:60]
        y_train_val = y_cls[:60]

        X_test = X_cls[60:]
        y_test = y_cls[60:]

        # 在 train_val 中隨機切出 validation
        idx = np.random.permutation(len(X_train_val))
        X_train_val = X_train_val[idx]
        y_train_val = y_train_val[idx]

        val_size = int(len(X_train_val) * val_ratio)

        X_val = X_train_val[:val_size]
        y_val = y_train_val[:val_size]

        X_train = X_train_val[val_size:]
        y_train = y_train_val[val_size:]

        X_train_list.append(X_train)
        y_train_list.append(y_train)

        X_val_list.append(X_val)
        y_val_list.append(y_val)

        X_test_list.append(X_test)
        y_test_list.append(y_test)

    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)

    X_val = np.vstack(X_val_list)
    y_val = np.hstack(y_val_list)

    X_test = np.vstack(X_test_list)
    y_test = np.hstack(y_test_list)

    # 整體再 shuffle train / val
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    perm = np.random.permutation(len(X_val))
    X_val = X_val[perm]
    y_val = y_val[perm]

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================================================
# 5. normalization
#    只能用 training set 的 mean / std
# =========================================================
def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Conv1d 輸入格式: (N, C, L)
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    return X_train, X_val, X_test, mean, std


# =========================================================
# 6. Dataset
# =========================================================
class SPCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# 7. 1D CNN 模型
# =========================================================
class CNN1D(nn.Module):
    def __init__(self, num_classes=6, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 8. 訓練一個 epoch
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_num += y.size(0)

    avg_loss = total_loss / total_num
    avg_acc = total_correct / total_num
    return avg_loss, avg_acc


# =========================================================
# 9. 驗證 / 測試
# =========================================================
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_num += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_num
    avg_acc = total_correct / total_num

    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


# =========================================================
# 10. 儲存 history.csv
# =========================================================
def save_history_csv(history, save_path):
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# =========================================================
# 11. 儲存 acc + loss 圖
#     同一張圖，上面 loss，下面 acc
# =========================================================
def save_acc_loss_curve(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Loss
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Validation Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(train_accs, label="Train Accuracy")
    axes[1].plot(val_accs, label="Validation Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# 12. 儲存混淆矩陣
# =========================================================
def save_confusion_matrix(y_true, y_pred, save_path):
    class_names = [
        "Normal",
        "Cyclic",
        "Increasing",
        "Decreasing",
        "Upward Shift",
        "Downward Shift"
    ]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=30, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================================================
# 13. 儲存訓練報告
# =========================================================
def save_training_report(
    save_path,
    run_dir,
    file_path,
    device,
    batch_size,
    learning_rate,
    num_epochs,
    val_ratio,
    weight_decay,
    dropout,
    optimizer_name,
    scheduler_name,
    loss_name,
    train_size,
    val_size,
    test_size,
    mean,
    std,
    best_val_acc,
    best_epoch,
    final_train_loss,
    final_train_acc,
    final_val_loss,
    final_val_acc,
    final_test_loss,
    final_test_acc,
    elapsed_time
):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("========== Training Report ==========\n")
        f.write(f"Run directory          : {run_dir}\n")
        f.write(f"Dataset file           : {file_path}\n")
        f.write(f"Device                 : {device}\n")
        f.write(f"Batch size             : {batch_size}\n")
        f.write(f"Learning rate          : {learning_rate}\n")
        f.write(f"Epochs                 : {num_epochs}\n")
        f.write(f"Validation ratio       : {val_ratio}\n")
        f.write(f"Weight decay           : {weight_decay}\n")
        f.write(f"Dropout                : {dropout}\n")
        f.write(f"Optimizer              : {optimizer_name}\n")
        f.write(f"Scheduler              : {scheduler_name}\n")
        f.write(f"Loss function          : {loss_name}\n")
        f.write(f"Normalization mean     : {mean:.6f}\n")
        f.write(f"Normalization std      : {std:.6f}\n")
        f.write(f"Elapsed time (sec)     : {elapsed_time:.2f}\n")
        f.write("\n")

        f.write("========== Dataset Info ==========\n")
        f.write("Total samples          : 600\n")
        f.write("Sequence length        : 60\n")
        f.write("Number of classes      : 6\n")
        f.write(f"Train samples          : {train_size}\n")
        f.write(f"Validation samples     : {val_size}\n")
        f.write(f"Test samples           : {test_size}\n")
        f.write("\n")

        f.write("========== Best Model ==========\n")
        f.write(f"Best Validation Acc    : {best_val_acc:.4f}\n")
        f.write(f"Best Epoch             : {best_epoch}\n")
        f.write("\n")

        f.write("========== Final Metrics ==========\n")
        f.write(f"Final Train Loss       : {final_train_loss:.4f}\n")
        f.write(f"Final Train Acc        : {final_train_acc:.4f}\n")
        f.write(f"Final Validation Loss  : {final_val_loss:.4f}\n")
        f.write(f"Final Validation Acc   : {final_val_acc:.4f}\n")
        f.write(f"Final Test Loss        : {final_test_loss:.4f}\n")
        f.write(f"Final Test Acc         : {final_test_acc:.4f}\n")
        f.write("\n")

        f.write("========== Model Summary ==========\n")
        f.write("Conv1d(1,16,3,padding=1) + BN + ReLU + MaxPool1d(2)\n")
        f.write("Conv1d(16,32,3,padding=1) + BN + ReLU + MaxPool1d(2)\n")
        f.write("Conv1d(32,64,3,padding=1) + BN + ReLU + AdaptiveAvgPool1d(1)\n")
        f.write("Flatten -> Linear(64,32) -> ReLU -> Dropout -> Linear(32,6)\n")


# =========================================================
# 14. 主程式
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    run_dir = create_output_dir()
    print(f"Output directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    start_time = time.time()

    # -----------------------------------------------------
    # 載入資料
    # -----------------------------------------------------
    X, y = load_data(FILE_PATH)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data_by_class(
        X, y, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )

    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)

    print("X_train shape:", X_train.shape)
    print("X_val shape  :", X_val.shape)
    print("X_test shape :", X_test.shape)

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------
    train_dataset = SPCDataset(X_train, y_train)
    val_dataset = SPCDataset(X_val, y_val)
    test_dataset = SPCDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------
    # 建立模型
    # -----------------------------------------------------
    model = CNN1D(num_classes=6, dropout=DROPOUT).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    # -----------------------------------------------------
    # 紀錄
    # -----------------------------------------------------
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    history = []

    best_val_acc = 0.0
    best_epoch = 0

    best_model_path = os.path.join(run_dir, "best_model.pth")

    # -----------------------------------------------------
    # 訓練
    # -----------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr
        })

        print(
            f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 用 validation accuracy 選 best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    # -----------------------------------------------------
    # 載入最佳模型，做最後 test
    # -----------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

    elapsed_time = time.time() - start_time

    # -----------------------------------------------------
    # 存圖與報告
    # -----------------------------------------------------
    save_acc_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        save_path=os.path.join(run_dir, "acc_loss_curve.png")
    )

    save_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        save_path=os.path.join(run_dir, "confusion_matrix.png")
    )

    save_history_csv(
        history=history,
        save_path=os.path.join(run_dir, "history.csv")
    )

    save_training_report(
        save_path=os.path.join(run_dir, "training_report.txt"),
        run_dir=run_dir,
        file_path=FILE_PATH,
        device=device,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        val_ratio=VAL_RATIO,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        optimizer_name="Adam",
        scheduler_name="StepLR(step_size=20, gamma=0.5)",
        loss_name="CrossEntropyLoss",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        mean=mean,
        std=std,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        final_train_loss=train_losses[-1],
        final_train_acc=train_accs[-1],
        final_val_loss=val_losses[-1],
        final_val_acc=val_accs[-1],
        final_test_loss=test_loss,
        final_test_acc=test_acc,
        elapsed_time=elapsed_time
    )

    # -----------------------------------------------------
    # 顯示最終結果
    # -----------------------------------------------------
    print("\n========== Training Finished ==========")
    print(f"Best Validation Accuracy : {best_val_acc:.4f}")
    print(f"Best Epoch               : {best_epoch}")
    print(f"Final Test Loss          : {test_loss:.4f}")
    print(f"Final Test Accuracy      : {test_acc:.4f}")
    print(f"All results saved to     : {run_dir}")


if __name__ == "__main__":
    main()