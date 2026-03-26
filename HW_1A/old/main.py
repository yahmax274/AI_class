from utils.read_img import load_cifar10_train, preview_cifar_images
if __name__ == "__main__":
    cifar_root = "datasets/cifar-10-batches-py"

    train_images, train_labels, label_names = load_cifar10_train(
        cifar_root=cifar_root,
        show_info=True
    )

    preview_cifar_images(
        images=train_images,
        labels=train_labels,
        label_names=label_names,
        num_images=12,
        cols=4,
        scale=4,
        use_matplotlib=True,
        save_path="cifar_preview.jpg"
    )