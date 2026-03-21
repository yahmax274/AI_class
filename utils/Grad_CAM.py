import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================================================
#  Grad-CAM
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

def get_gradcam_target_layer(model):
    """
    根據模型類型回傳適合的 Grad-CAM target layer
    """
    if hasattr(model, "features"):
        # VGG
        return model.features[30]
    elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        # ResNet50
        return model.backbone.layer4[-1].conv3
    else:
        raise ValueError("無法辨識模型的 Grad-CAM target layer")

def predict_and_visualize_samples(
    model,
    images,
    labels,
    label_names,
    transform,
    device,
    indices,
    save_dir="resnet50_predict_results"
):
    os.makedirs(save_dir, exist_ok=True)

    target_layer = get_gradcam_target_layer(model)
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

        # print(f"[Saved] {save_path} | GT={label_names[gt_idx]} | Pred={label_names[pred_idx]} | Conf={confidence:.4f}")

    gradcam.remove_hooks()

# =========================================================
#  Prediction + Heatmap 視覺化工具
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

    plt.subplot(1, 2, 1)
    plt.imshow(annotated_img)
    plt.axis("off")
    plt.title("Original + Prediction")

    # plt.subplot(1, 3, 2)
    # plt.imshow(heatmap_rgb)
    # plt.axis("off")
    # plt.title("Grad-CAM Heatmap")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Overlay ({result_text})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()