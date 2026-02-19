# eval.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import f1_score, confusion_matrix, classification_report

from dvm_color_dataset import build_samples, make_splits, DVMColorDataset
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
def predict(model, loader, device):
    model.eval()
    preds_all = []
    tgts_all = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu()
        preds_all.append(preds)
        tgts_all.append(yb)
    return torch.cat(preds_all).numpy(), torch.cat(tgts_all).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="Confirmed_fronts")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    meta = ckpt["meta"]
    arch = meta["arch"]
    class_to_idx = meta["class_to_idx"]
    idx_to_class = {int(k): v for k, v in meta["idx_to_class"].items()} if isinstance(next(iter(meta["idx_to_class"])), str) else meta["idx_to_class"]
    num_classes = meta["num_classes"]
    img_size = meta.get("img_size", 224)

    samples = build_samples(args.data_root)
    split = make_splits(samples, val_size=0.15, test_size=0.15, seed=args.seed, min_count_per_class=meta.get("min_count_per_class", 5))

    tfms = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_ds = DVMColorDataset(split.test, class_to_idx, transform=tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_model(arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    preds, tgts = predict(model, test_loader, device)
    f1 = f1_score(tgts, preds, average="macro")
    print("arch:", arch)
    print("F1_macro:", f1)

    labels = list(range(num_classes))
    target_names = [idx_to_class[i] for i in labels if i in idx_to_class]
    report = classification_report(tgts, preds, labels=labels, target_names=target_names, zero_division=0)
    print(report)

    cm = confusion_matrix(tgts, preds, labels=labels)
    out_dir = Path(args.ckpt).parent
    (out_dir / "f1.json").write_text(json.dumps({"arch": arch, "f1_macro": float(f1)}, indent=2), encoding="utf-8")
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("Saved: f1.json, confusion_matrix.json, classification_report.txt ->", out_dir)


if __name__ == "__main__":
    main()
