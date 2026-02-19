# infer.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

from models_resnet18_scratch import resnet18_scratch


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18_scratch":
        return resnet18_scratch(num_classes)
    if arch == "resnet50_ft":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "efficientnet_b0_ft":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt["meta"]
    arch = meta["arch"]
    num_classes = meta["num_classes"]
    idx_to_class = meta["idx_to_class"]
    img_size = meta.get("img_size", 224)

    model = build_model(arch, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)

    logits = model(x)
    pred = int(logits.argmax(dim=1).item())

    # idx_to_class может быть с ключами-строками
    label = idx_to_class.get(str(pred), idx_to_class.get(pred, "UNKNOWN"))

    print("arch:", arch)
    print("image:", Path(args.image).name)
    print("pred_class_idx:", pred)
    print("pred_color:", label)


if __name__ == "__main__":
    main()
