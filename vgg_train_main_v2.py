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

# =========================================================
# 1. CIFAR-10 Dataset
# =========================================================
class CIFAR10ArrayDataset(torch.utils.data.Dataset):
    """
    將已讀入記憶體中的 CIFAR-10 影像與標籤包成 PyTorch Dataset
    images: [N, 32, 32, 3], RGB, uint8
    labels: [N]
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]   # numpy, [32, 32, 3], RGB
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


# =========================================================
# 2. VGG16_BN for CIFAR-10
# =========================================================
class VGG16_BN_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),   # 0
            nn.BatchNorm2d(64),                                        # 1
            nn.ReLU(inplace=True),                                     # 2

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # 3
            nn.BatchNorm2d(64),                                        # 4
            nn.ReLU(inplace=True),                                     # 5

            nn.MaxPool2d(kernel_size=2, stride=2),                     # 6

            # Block 2: 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), # 7
            nn.BatchNorm2d(128),                                       # 8
            nn.ReLU(inplace=True),                                     # 9

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),# 10
            nn.BatchNorm2d(128),                                       # 11
            nn.ReLU(inplace=True),                                     # 12

            nn.MaxPool2d(kernel_size=2, stride=2),                     # 13

            # Block 3: 8 -> 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),# 14
            nn.BatchNorm2d(256),                                       # 15
            nn.ReLU(inplace=True),                                     # 16

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),# 17
            nn.BatchNorm2d(256),                                       # 18
            nn.ReLU(inplace=True),                                     # 19

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),# 20
            nn.BatchNorm2d(256),                                       # 21
            nn.ReLU(inplace=True),                                     # 22

            nn.MaxPool2d(kernel_size=2, stride=2),                     # 23

            # Block 4: 4 -> 2
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),# 24
            nn.BatchNorm2d(512),                                       # 25
            nn.ReLU(inplace=True),                                     # 26

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),# 27
            nn.BatchNorm2d(512),                                       # 28
            nn.ReLU(inplace=True),                                     # 29

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),# 30
            nn.BatchNorm2d(512),                                       # 31
            nn.ReLU(inplace=True),                                     # 32

            nn.MaxPool2d(kernel_size=2, stride=2),                     # 33

            # Block 5: 2 -> 1
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),# 34
            nn.BatchNorm2d(512),                                       # 35
            nn.ReLU(inplace=True),                                     # 36

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),# 37
            nn.BatchNorm2d(512),                                       # 38
            nn.ReLU(inplace=True),                                     # 39

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),# 40
            nn.BatchNorm2d(512),                                       # 41
            nn.ReLU(inplace=True),                                     # 42

            nn.MaxPool2d(kernel_size=2, stride=2)                      # 43
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

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
# 3. CIFAR-10 原始 batch 讀取
# =========================================================
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_cifar10_label_names(cifar_root, show_info=False):
    meta_path = os.path.join(cifar_root, "batches.meta")
    meta = load_cifar_batch(meta_path)
    label_names = [name.decode("utf-8") for name in meta[b'label_names']]

    if show_info:
        print("CIFAR-10 類別名稱：")
        print(label_names)

    return label_names


def load_cifar10_train(cifar_root, show_info=False):
    label_names = load_cifar10_label_names(cifar_root, show_info=show_info)

    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = os.path.join(cifar_root, f"data_batch_{i}")
        batch = load_cifar_batch(batch_path)

        data = batch[b'data']
        labels = batch[b'labels']

        train_data_list.append(data)
        train_labels_list.extend(labels)

        if show_info:
            print(f"已讀取: data_batch_{i}, data shape = {data.shape}, labels = {len(labels)}")

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.array(train_labels_list)
    train_images = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    if show_info:
        print("\n訓練資料讀取完成")
        print("train_data shape:", train_data.shape)
        print("train_labels shape:", train_labels.shape)
        print("train_images shape:", train_images.shape)

    return train_images, train_labels, label_names


def load_cifar10_test(cifar_root, show_info=False):
    label_names = load_cifar10_label_names(cifar_root, show_info=False)

    batch_path = os.path.join(cifar_root, "test_batch")
    batch = load_cifar_batch(batch_path)

    test_data = batch[b'data']
    test_labels = np.array(batch[b'labels'])
    test_images = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    if show_info:
        print("\n測試資料讀取完成")
        print("test_data shape:", test_data.shape)
        print("test_labels shape:", test_labels.shape)
        print("test_images shape:", test_images.shape)

    return test_images, test_labels, label_names


# =========================================================
# 4. 訓練 / 評估
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

@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix_sklearn(
    y_true,
    y_pred,
    class_names,
    save_path="confusion_matrix.png",
    normalize=None,
    title="Confusion Matrix",
    cmap="Blues"
):
    """
    normalize:
        None   -> 顯示 raw counts
        "true" -> 每一列正規化
        "pred" -> 每一行正規化
        "all"  -> 全域正規化
    """
    labels = np.arange(len(class_names))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(
        ax=ax,
        cmap=cmap,
        xticks_rotation=45,
        colorbar=True,
        values_format="d"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"混淆矩陣已儲存: {save_path}")
    print("\nConfusion Matrix (raw counts):")
    print(cm)

    if normalize is not None:
        cm_norm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize=normalize
        )

        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)

        norm_save_path = save_path.replace(".png", f"_norm_{normalize}.png")

        fig, ax = plt.subplots(figsize=(10, 8))
        disp_norm.plot(
            ax=ax,
            cmap=cmap,
            xticks_rotation=45,
            colorbar=True,
            values_format=".2f"
        )
        ax.set_title(f"{title} (normalize={normalize})")
        plt.tight_layout()
        plt.savefig(norm_save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"正規化混淆矩陣已儲存: {norm_save_path}")
        print(f"\nConfusion Matrix (normalize={normalize}):")
        print(cm_norm)

    return cm

def print_classification_report_sklearn(y_true, y_pred, class_names):
    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

# =========================================================
# 5. 畫圖
# =========================================================
def plot_curves(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_prefix="VGG16_BN"):
    epochs = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    save_name = f"{title_prefix.lower()}_curve.png"
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"曲線圖已儲存為 {save_name}")
    plt.close()


# =========================================================
# 6. Grad-CAM
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: [1, 3, H, W]
        回傳:
            cam: [H, W], 0~1
            logits: [1, num_classes]
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # 權重 = gradient 的 global average pooling
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 加權 activation
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # 上採樣回輸入大小
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        # normalize 到 0~1
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam, logits.detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# =========================================================
# 7. Prediction + Heatmap 視覺化工具
# =========================================================
def create_heatmap_overlay(image_rgb, cam, vis_size=256, alpha=0.4):
    """
    image_rgb: [H, W, 3], uint8
    cam: [H, W], 0~1
    """
    image_vis = cv2.resize(image_rgb, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)

    cam_uint8 = np.uint8(cam * 255)
    cam_uint8 = cv2.resize(cam_uint8, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_vis, 1 - alpha, heatmap_rgb, alpha, 0)

    return image_vis, heatmap_rgb, overlay


def draw_prediction_text(image_rgb, gt_idx, pred_idx, confidence, label_names, vis_size=256):
    """
    在原圖上印出 GT / Pred / Confidence / 對錯
    """
    image_vis = cv2.resize(image_rgb, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
    image_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    is_correct = (gt_idx == pred_idx)
    result_text = "Correct" if is_correct else "Wrong"
    color = (0, 255, 0) if is_correct else (0, 0, 255)

    text1 = f"GT: {label_names[gt_idx]}"
    text2 = f"Pred: {label_names[pred_idx]} ({confidence * 100:.1f}%)"
    text3 = result_text

    cv2.putText(image_bgr, text1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image_bgr, text2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(image_bgr, text3, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, text3, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_prediction_with_heatmap(
    image_rgb,
    gt_idx,
    pred_idx,
    confidence,
    cam,
    label_names,
    save_path,
    vis_size=256
):
    annotated_img = draw_prediction_text(
        image_rgb=image_rgb,
        gt_idx=gt_idx,
        pred_idx=pred_idx,
        confidence=confidence,
        label_names=label_names,
        vis_size=vis_size
    )

    _, heatmap_rgb, overlay = create_heatmap_overlay(
        image_rgb=image_rgb,
        cam=cam,
        vis_size=vis_size,
        alpha=0.4
    )

    result_text = "Correct" if gt_idx == pred_idx else "Wrong"

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(annotated_img)
    plt.axis("off")
    plt.title("Original + Prediction")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Overlay ({result_text})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def find_misclassified_indices(model, images, labels, transform, device, max_samples=8):
    """
    找出錯誤分類的 sample index
    """
    model.eval()
    wrong_indices = []

    with torch.no_grad():
        for idx in range(len(images)):
            input_tensor = transform(images[idx]).unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred_idx = int(logits.argmax(dim=1).item())
            gt_idx = int(labels[idx])

            if pred_idx != gt_idx:
                wrong_indices.append(idx)

            if len(wrong_indices) >= max_samples:
                break

    return wrong_indices

def find_correct_indices_per_class(
    model,
    images,
    labels,
    transform,
    device,
    num_classes,
    max_samples_per_class=1,
    label_names=None
):
    """
    找出每個類別中，分類正確的 sample index
    每個類別最多抓 max_samples_per_class 張
    """
    model.eval()

    correct_dict = {class_idx: [] for class_idx in range(num_classes)}

    with torch.no_grad():
        for idx in range(len(images)):
            gt_idx = int(labels[idx])

            # 如果這個類別已經收集夠了，就跳過
            if len(correct_dict[gt_idx]) >= max_samples_per_class:
                continue

            input_tensor = transform(images[idx]).unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred_idx = int(logits.argmax(dim=1).item())

            if pred_idx == gt_idx:
                correct_dict[gt_idx].append(idx)

            # 若所有類別都已蒐集完成，就提早停止
            done = all(len(correct_dict[c]) >= max_samples_per_class for c in range(num_classes))
            if done:
                break

    # 印出每個類別找到的正確樣本
    print("\n每個類別正確分類的 index：")
    for class_idx in range(num_classes):
        class_name = label_names[class_idx] if label_names is not None else str(class_idx)
        print(f"  Class {class_idx} ({class_name}): {correct_dict[class_idx]}")

    # 攤平成一個 list，方便後續直接視覺化
    correct_indices = []
    for class_idx in range(num_classes):
        correct_indices.extend(correct_dict[class_idx])

    return correct_indices, correct_dict

def predict_and_visualize_samples(
    model,
    images,
    labels,
    label_names,
    transform,
    device,
    indices,
    save_dir="vgg16_bn_predict_results"
):
    os.makedirs(save_dir, exist_ok=True)

    # CIFAR-10 很小，建議抓 Block4 最後一個 Conv
    # index 30 對應 Block4 的最後一個 Conv2d，特徵圖大小約 4x4
    target_layer = model.features[30]

    gradcam = GradCAM(model, target_layer)

    for idx in indices:
        image_rgb = images[idx]
        gt_idx = int(labels[idx])

        input_tensor = transform(image_rgb).unsqueeze(0).to(device)

        cam, logits = gradcam.generate(input_tensor=input_tensor, class_idx=None)

        probs = F.softmax(logits, dim=1)
        confidence, pred_idx_tensor = probs.max(dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = float(confidence.item())

        result_flag = "correct" if gt_idx == pred_idx else "wrong"
        result_name = f"{result_flag}_idx_{idx:04d}_gt_{label_names[gt_idx]}_pred_{label_names[pred_idx]}.png"
        save_path = os.path.join(save_dir, result_name)

        save_prediction_with_heatmap(
            image_rgb=image_rgb,
            gt_idx=gt_idx,
            pred_idx=pred_idx,
            confidence=confidence,
            cam=cam,
            label_names=label_names,
            save_path=save_path,
            vis_size=256
        )

        print(f"[Saved] {save_path} | GT={label_names[gt_idx]} | Pred={label_names[pred_idx]} | Conf={confidence:.4f}")

    gradcam.remove_hooks()


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
    num_epochs = 100

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
    num_correct_per_class = 10

    # 輸出路徑
    best_model_path = "best_vgg16_bn_cifar10.pth"
    predict_save_dir = "vgg16_bn_predict_results"
    confusion_matrix_path = "vgg16_bn_confusion_matrix.png"

    # 使用裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)

    # 固定亂數種子
    torch.manual_seed(42)
    np.random.seed(42)

    # -----------------------------
    # 讀取資料
    # -----------------------------
    train_images, train_labels, label_names = load_cifar10_train(
        cifar_root=cifar_root,
        show_info=True
    )

    test_images, test_labels, _ = load_cifar10_test(
        cifar_root=cifar_root,
        show_info=True
    )

    # -----------------------------
    # Transform
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # -----------------------------
    # Dataset / DataLoader
    # -----------------------------
    train_dataset = CIFAR10ArrayDataset(
        images=train_images,
        labels=train_labels,
        transform=train_transform
    )

    test_dataset = CIFAR10ArrayDataset(
        images=test_images,
        labels=test_labels,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # -----------------------------
    # 建立模型
    # -----------------------------
    model = VGG16_BN_CIFAR10(num_classes=num_classes).to(device)

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

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.1
        )

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

        plot_curves(
            train_loss_list=train_loss_list,
            train_acc_list=train_acc_list,
            test_loss_list=test_loss_list,
            test_acc_list=test_acc_list,
            title_prefix="VGG16_BN"
        )

        print(f"\n訓練完成，最佳測試準確率: {best_test_acc:.4f}，出現在 Epoch {best_epoch}")

    # -----------------------------
    # Prediction + Grad-CAM
    # -----------------------------
    if DO_PREDICT:
        print("\n開始載入最佳模型進行 prediction 與 heatmap 視覺化...")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"找不到最佳模型檔案: {best_model_path}")

        best_model = VGG16_BN_CIFAR10(num_classes=num_classes).to(device)
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
