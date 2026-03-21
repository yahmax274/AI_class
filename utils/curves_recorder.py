import matplotlib.pyplot as plt
# =========================================================
#  畫圖
# =========================================================
def plot_curves(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_prefix="Model"):
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