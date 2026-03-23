import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import OrderedDict
from pathlib import Path
from contextlib import contextmanager

from utils.load_CIFAR10_data_v2 import build_cifar10_dataloaders
from utils.create_confusion_matrix import collect_predictions, plot_confusion_matrix_sklearn, print_classification_report_sklearn
from utils.Grad_CAM import predict_and_visualize_samples
from utils.classified_indices import find_misclassified_indices, find_correct_indices_per_class
from utils.curves_recorder import plot_curves

# =========================================================
#  VGG16_BN for CIFAR-10
# =========================================================
class VGG16_BN_CIFAR10_Better(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            # =========================
            # Block 1: 32 -> 16
            # =========================
            ("block1_conv1", nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
            ("block1_bn1", nn.BatchNorm2d(64)),
            ("block1_relu1", nn.ReLU(inplace=True)),

            ("block1_conv2", nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)),
            ("block1_bn2", nn.BatchNorm2d(64)),
            ("block1_relu2", nn.ReLU(inplace=True)),

            ("block1_pool", nn.MaxPool2d(kernel_size=2, stride=2)),

            # =========================
            # Block 2: 16 -> 8
            # =========================
            ("block2_conv1", nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)),
            ("block2_bn1", nn.BatchNorm2d(128)),
            ("block2_relu1", nn.ReLU(inplace=True)),

            ("block2_conv2", nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)),
            ("block2_bn2", nn.BatchNorm2d(128)),
            ("block2_relu2", nn.ReLU(inplace=True)),

            ("block2_pool", nn.MaxPool2d(kernel_size=2, stride=2)),

            # =========================
            # Block 3: 8 -> 4
            # =========================
            ("block3_conv1", nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)),
            ("block3_bn1", nn.BatchNorm2d(256)),
            ("block3_relu1", nn.ReLU(inplace=True)),

            ("block3_conv2", nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ("block3_bn2", nn.BatchNorm2d(256)),
            ("block3_relu2", nn.ReLU(inplace=True)),

            ("block3_conv3", nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)),
            ("block3_bn3", nn.BatchNorm2d(256)),
            ("block3_relu3", nn.ReLU(inplace=True)),

            ("block3_pool", nn.MaxPool2d(kernel_size=2, stride=2)),

            # =========================
            # Block 4: 4 -> 2
            # 原本 512 改成 384
            # =========================
            ("block4_conv1", nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False)),
            ("block4_bn1", nn.BatchNorm2d(384)),
            ("block4_relu1", nn.ReLU(inplace=True)),

            ("block4_conv2", nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ("block4_bn2", nn.BatchNorm2d(384)),
            ("block4_relu2", nn.ReLU(inplace=True)),

            ("block4_conv3", nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ("block4_bn3", nn.BatchNorm2d(384)),
            ("block4_relu3", nn.ReLU(inplace=True)),

            ("block4_pool", nn.MaxPool2d(kernel_size=2, stride=2)),

            # =========================
            # Block 5: 2 -> 1
            # 原本 512 改成 384
            # =========================
            ("block5_conv1", nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ("block5_bn1", nn.BatchNorm2d(384)),
            ("block5_relu1", nn.ReLU(inplace=True)),

            ("block5_conv2", nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ("block5_bn2", nn.BatchNorm2d(384)),
            ("block5_relu2", nn.ReLU(inplace=True)),

            ("block5_conv3", nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ("block5_bn3", nn.BatchNorm2d(384)),
            ("block5_relu3", nn.ReLU(inplace=True)),

            ("block5_pool", nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(384, 256)),
            ("relu1", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(dropout)),
            ("fc2", nn.Linear(256, num_classes)),
        ]))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

# =========================================================
#  訓練 / 評估
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    start_time = time.time()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_acc, epoch_time


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

def prepare_output_dirs(output_root):
    """
    建立此次實驗的母資料夾與子資料夾
    """
    output_root = Path(output_root)

    paths = {
        "root": output_root,
        "models": output_root / "models",
        "predictions": output_root / "predictions",
        "figures": output_root / "figures",
        "reports": output_root / "reports",
    }

    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths

def save_best_accuracy_report(report_path, best_acc, best_epoch, best_model_path):
    """
    將最佳測試準確率寫入 reports 資料夾
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Best Training Report\n")
        f.write("====================\n")
        f.write(f"Best Test Accuracy : {best_acc:.4f}\n")
        f.write(f"Best Epoch         : {best_epoch}\n")
        f.write(f"Best Model Path    : {best_model_path}\n")

@contextmanager
def temp_chdir(path):
    """
    暫時切換工作目錄，讓某些沒有提供 save_path 的 utility
    也能把結果存進指定資料夾
    """
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


# =========================================================
# 8. 主程式
# =========================================================
if __name__ == "__main__":
    # -----------------------------
    # 基本設定
    # -----------------------------
    cifar_root = "datasets/cifar-10-batches-py"
    batch_size = 64
    num_workers = 0
    num_classes = 10
    num_epochs = 300
    # num_epochs = 80

    # 功能控制
    DO_TRAIN = True
    DO_PREDICT = True

    # prediction 模式
    # 可選: "manual"、"misclassified"、"correct_per_class"、"both"
    PREDICT_MODE = "both"

    # 手動指定想看的 test index
    manual_indices = [0, 1, 2, 3, 4, 5, 10, 20]

    # 自動找幾張錯誤分類
    num_wrong_cases = 10

    # 每個類別要抓幾張正確分類樣本
    num_correct_per_class = 5

    # 輸出路徑
    OUTPUT_ROOT = "outputs/my_vgg16_experiment/run4"

    save_dirs = prepare_output_dirs(OUTPUT_ROOT)

    best_model_path = str(save_dirs["models"] / "best_vgg16_bn_cifar10.pth")
    predict_save_dir = str(save_dirs["predictions"])
    confusion_matrix_path = str(save_dirs["figures"] / "vgg16_bn_confusion_matrix.png")
    best_report_path = str(save_dirs["reports"] / "best_accuracy_report.txt")

    # 使用裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)

    # 固定亂數種子
    torch.manual_seed(42)
    np.random.seed(42)

    data_dict = build_cifar10_dataloaders(
        cifar_root=cifar_root,
        batch_size=batch_size,
        num_workers=num_workers
    )

    train_images = data_dict["train_images"]
    train_labels = data_dict["train_labels"]
    test_images = data_dict["test_images"]
    test_labels = data_dict["test_labels"]
    label_names = data_dict["label_names"]
    train_loader = data_dict["train_loader"]
    test_loader = data_dict["test_loader"]
    test_transform = data_dict["test_transform"]

    # -----------------------------
    # 建立模型
    # -----------------------------
    # model = VGG16_BN_CIFAR10(num_classes=num_classes).to(device)
    model = VGG16_BN_CIFAR10_Better(num_classes=num_classes, dropout=0.3).to(device)

    # forward 檢查
    images, labels = next(iter(train_loader))
    images = images.to(device)
    outputs = model(images)
    print("input shape :", images.shape)
    print("output shape:", outputs.shape)

    # -----------------------------
    # 訓練
    # -----------------------------
    if DO_TRAIN:
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=1e-3,           
                    weight_decay=0.05, 
                    betas=(0.9, 0.999))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer= optimizer, T_max=num_epochs)

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        train_time_list = []

        best_test_acc = -1.0
        best_epoch = -1

        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc, train_time = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            test_loss, test_acc = evaluate(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device
            )

            scheduler.step()

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            train_time_list.append(train_time)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"已儲存最佳模型: {best_model_path}")

            print(f"Epoch [{epoch+1}/{num_epochs}] | lr={current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
            print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}, Best Test Acc: {best_test_acc:.4f} (Epoch {best_epoch})")

        with temp_chdir(save_dirs["figures"]):
            plot_curves(
                train_loss_list=train_loss_list,
                train_acc_list=train_acc_list,
                test_loss_list=test_loss_list,
                test_acc_list=test_acc_list,
                title_prefix="VGG16_BN"
            )

        save_best_accuracy_report(
            report_path=best_report_path,
            best_acc=best_test_acc,
            best_epoch=best_epoch,
            best_model_path=best_model_path
        )

        print(f"\n訓練完成，最佳測試準確率: {best_test_acc:.4f}，出現在 Epoch {best_epoch}")
        print(f"最佳準確率報告已儲存至: {best_report_path}")

    # -----------------------------
    # Prediction + Grad-CAM
    # -----------------------------
    if DO_PREDICT:
        print("\n開始載入最佳模型進行 prediction 與 heatmap 視覺化...")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"找不到最佳模型檔案: {best_model_path}")

        # best_model = VGG16_BN_CIFAR10(num_classes=num_classes).to(device)
        best_model = VGG16_BN_CIFAR10_Better(num_classes=num_classes, dropout=0.3).to(device)
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_model.eval()

        # =============================
        # sklearn 混淆矩陣與分類報告
        # =============================
        y_true, y_pred = collect_predictions(
            model=best_model,
            dataloader=test_loader,
            device=device
        )

        cm = plot_confusion_matrix_sklearn(
            y_true=y_true,
            y_pred=y_pred,
            class_names=label_names,
            save_path=confusion_matrix_path,
            normalize="true",   # 可改成 None / "true" / "pred" / "all"
            title="VGG16_BN CIFAR-10 Confusion Matrix"
        )

        print_classification_report_sklearn(
            y_true=y_true,
            y_pred=y_pred,
            class_names=label_names
        )

        if PREDICT_MODE == "manual":
            target_indices = manual_indices

        elif PREDICT_MODE == "misclassified":
            wrong_indices = find_misclassified_indices(
                model=best_model,
                images=test_images,
                labels=test_labels,
                transform=test_transform,
                device=device,
                max_samples=num_wrong_cases
            )
            print("找到的錯誤分類 index:", wrong_indices)
            target_indices = wrong_indices

        elif PREDICT_MODE == "correct_per_class":
            correct_indices, correct_dict = find_correct_indices_per_class(
                model=best_model,
                images=test_images,
                labels=test_labels,
                transform=test_transform,
                device=device,
                num_classes=num_classes,
                max_samples_per_class=num_correct_per_class,
                label_names=label_names
            )
            target_indices = correct_indices

        elif PREDICT_MODE == "both":
            wrong_indices = find_misclassified_indices(
                model=best_model,
                images=test_images,
                labels=test_labels,
                transform=test_transform,
                device=device,
                max_samples=num_wrong_cases
            )
            print("找到的錯誤分類 index:", wrong_indices)

            correct_indices, correct_dict = find_correct_indices_per_class(
                model=best_model,
                images=test_images,
                labels=test_labels,
                transform=test_transform,
                device=device,
                num_classes=num_classes,
                max_samples_per_class=num_correct_per_class,
                label_names=label_names
            )

            # 合併並去重複
            target_indices = wrong_indices + correct_indices
            target_indices = list(dict.fromkeys(target_indices))

            print("最終要視覺化的 index:", target_indices)

        else:
            raise ValueError("PREDICT_MODE 只能是 'manual'、'misclassified'、'correct_per_class'、'both'")

        if len(target_indices) == 0:
            print("沒有找到符合條件的樣本，略過視覺化。")
        else:
            predict_and_visualize_samples(
                model=best_model,
                images=test_images,
                labels=test_labels,
                label_names=label_names,
                transform=test_transform,
                device=device,
                indices=target_indices,
                save_dir=predict_save_dir
            )

            print(f"\nPrediction 完成，結果已儲存至資料夾: {predict_save_dir}")
