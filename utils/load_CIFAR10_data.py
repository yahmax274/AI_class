import torch
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms

# =========================================================
# CIFAR-10 Dataset
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
#  CIFAR-10 原始 batch 讀取
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

def build_cifar10_dataloaders(cifar_root, batch_size=64, num_workers=0):
    train_images, train_labels, label_names = load_cifar10_train(
        cifar_root=cifar_root,
        show_info=True
    )

    test_images, test_labels, _ = load_cifar10_test(
        cifar_root=cifar_root,
        show_info=True
    )

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

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "label_names": label_names,
        "train_transform": train_transform,
        "test_transform": test_transform,
        "train_loader": train_loader,
        "test_loader": test_loader
    }