import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


class CIFAR10ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


class VGG16_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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


if __name__ == "__main__":
    cifar_root = "datasets/cifar-10-batches-py"

    train_images, train_labels, label_names = load_cifar10_train(
        cifar_root=cifar_root,
        show_info=True
    )

    test_images, test_labels, _ = load_cifar10_test(
        cifar_root=cifar_root,
        show_info=True
    )

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
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
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_CIFAR10(num_classes=10).to(device)

    images, labels = next(iter(train_loader))
    images = images.to(device)
    outputs = model(images)
    print("input shape :", images.shape)
    print("output shape:", outputs.shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    train_time_list = []

    for epoch in range(num_epochs):
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

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        train_time_list.append(train_time)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VGG16 Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("VGG16 Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()