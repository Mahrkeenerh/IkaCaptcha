"""Shared image preprocessing for inference."""

import numpy as np
from PIL import Image
from torchvision import transforms

from ikaptcha.model import IMG_H, IMG_W

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# PyTorch transform: PIL.Image -> normalized FloatTensor (3, H, W)
val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Pure-numpy preprocessing for ONNX inference. Returns (1, 3, H, W) float32."""
    img = img.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr[None, ...].astype(np.float32)
