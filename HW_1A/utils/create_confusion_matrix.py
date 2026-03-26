import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

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