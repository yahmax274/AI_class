import torch
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
    # print("\n每個類別正確分類的 index：")
    for class_idx in range(num_classes):
        class_name = label_names[class_idx] if label_names is not None else str(class_idx)
        # print(f"  Class {class_idx} ({class_name}): {correct_dict[class_idx]}")

    # 攤平成一個 list，方便後續直接視覺化
    correct_indices = []
    for class_idx in range(num_classes):
        correct_indices.extend(correct_dict[class_idx])

    return correct_indices, correct_dict