import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_cifar_batch(file_path):
    """
    讀取單一 CIFAR-10 batch 檔案
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_cifar10_label_names(cifar_root, show_info=False):
    """
    讀取 CIFAR-10 類別名稱

    Args:
        cifar_root (str): CIFAR-10 資料夾路徑
        show_info (bool): 是否顯示讀取資訊，預設 False

    Returns:
        list[str]: 類別名稱列表
    """
    meta_path = os.path.join(cifar_root, "batches.meta")
    meta = load_cifar_batch(meta_path)

    label_names = [name.decode("utf-8") for name in meta[b'label_names']]

    if show_info:
        print("CIFAR-10 類別名稱：")
        print(label_names)

    return label_names


def load_cifar10_train(cifar_root, show_info=False):
    """
    讀取 CIFAR-10 的完整訓練集（data_batch_1 ~ data_batch_5）

    Args:
        cifar_root (str): CIFAR-10 資料夾路徑
        show_info (bool): 是否顯示讀取資訊，預設 False

    Returns:
        train_images (np.ndarray): shape = [50000, 32, 32, 3]，RGB 格式
        train_labels (np.ndarray): shape = [50000]
        label_names (list[str]): 類別名稱
    """
    label_names = load_cifar10_label_names(cifar_root, show_info=show_info)

    train_data_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = os.path.join(cifar_root, f"data_batch_{i}")
        batch = load_cifar_batch(batch_path)

        data = batch[b'data']      # [10000, 3072]
        labels = batch[b'labels']  # list of length 10000

        train_data_list.append(data)
        train_labels_list.extend(labels)

        if show_info:
            print(f"已讀取: data_batch_{i}, data shape = {data.shape}, labels = {len(labels)}")

    train_data = np.concatenate(train_data_list, axis=0)   # [50000, 3072]
    train_labels = np.array(train_labels_list)             # [50000]

    # 還原成影像格式
    train_images = train_data.reshape(-1, 3, 32, 32)       # [N, 3, 32, 32]
    train_images = train_images.transpose(0, 2, 3, 1)      # [N, 32, 32, 3]，RGB

    if show_info:
        print("\n訓練資料讀取完成")
        print("train_data shape:", train_data.shape)
        print("train_labels shape:", train_labels.shape)
        print("train_images shape:", train_images.shape)

    return train_images, train_labels, label_names


def preview_cifar_images(images, labels, label_names, num_images=12, cols=4, scale=4,
                         window_name="CIFAR-10 Preview", use_matplotlib=True, save_path=None):
    num_images = min(num_images, len(images))
    rows = int(np.ceil(num_images / cols))

    cell_h = 32 * scale
    cell_w = 32 * scale
    text_h = 25

    canvas_h = rows * (cell_h + text_h)
    canvas_w = cols * cell_w

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i in range(num_images):
        row = i // cols
        col = i % cols

        img = images[i]  # RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)

        y1 = row * (cell_h + text_h)
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w

        canvas[y1:y2, x1:x2] = img_bgr

        label_text = label_names[labels[i]]
        cv2.putText(
            canvas,
            label_text,
            (x1 + 2, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    if save_path is not None:
        cv2.imwrite(save_path, canvas)
        print(f"預覽圖已儲存至: {save_path}")

    if use_matplotlib:
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(canvas_rgb)
        plt.axis("off")
        plt.title(window_name)
        plt.show()