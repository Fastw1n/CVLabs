# train.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

from dvm_color_dataset import build_samples, make_splits, DVMColorDataset
from models_resnet18_scratch import resnet18_scratch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
def eval_acc(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        total += yb.numel()
        correct += (preds == yb).sum().item()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, required=True,
                    choices=["resnet18_scratch", "resnet50_ft", "efficientnet_b0_ft"])
    ap.add_argument("--data_root", type=str, default="Confirmed_fronts")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--min_count_per_class", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = Path(args.out_dir) / args.arch
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args.data_root)
    split = make_splits(samples, val_size=0.15, test_size=0.15, seed=args.seed, min_count_per_class=args.min_count_per_class)
    num_classes = len(split.class_to_idx)
    print("num_classes:", num_classes)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = DVMColorDataset(split.train, split.class_to_idx, transform=train_tfms)
    val_ds   = DVMColorDataset(split.val,   split.class_to_idx, transform=test_tfms)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model = build_model(args.arch, num_classes=num_classes).to(device)

    # loss/opt
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    meta = {
        "arch": args.arch,
        "num_classes": num_classes,
        "class_to_idx": split.class_to_idx,
        "idx_to_class": split.idx_to_class,
        "data_root": args.data_root,
        "img_size": args.img_size,
        "seed": args.seed,
        "min_count_per_class": args.min_count_per_class,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    best_val_acc = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            seen += yb.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, seen)
        val_acc = eval_acc(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "meta": meta}, best_path)
            print(f"  saved: {best_path}")

    print("Done. Best val_acc:", best_val_acc)


if __name__ == "__main__":
    main()
