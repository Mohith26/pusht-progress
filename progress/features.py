"""Frozen ResNet-18 (ImageNet) features, standing in for the VLM backbone."""

import numpy as np
import torch
import torchvision


def resnet18_features(images, batch_size: int = 256, size: int = 160) -> np.ndarray:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = np.stack(images[start : start + batch_size]).astype(np.float32) / 255.0
            x = torch.from_numpy(batch).permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            out.append(net((x - mean) / std).numpy())
    return np.concatenate(out).astype(np.float32)


def build_inputs(feats: np.ndarray, first_idx: np.ndarray, mode: str) -> np.ndarray:
    """mode='abs' -> [f_t]; mode='delta' -> [f_0, f_t, f_t - f_0] as in the post."""
    ft = feats
    if mode == "abs":
        return ft
    if mode == "delta":
        f0 = feats[first_idx]
        return np.concatenate([f0, ft, ft - f0], axis=1)
    raise ValueError(f"unknown mode {mode!r}")
