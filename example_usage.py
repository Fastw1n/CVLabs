# пример использования: src/datasets/example_usage.py
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dvm_color_dataset import build_samples, make_splits, DVMColorDataset


def main():
    root = Path("Confirmed_fronts")  # <-- путь к распакованной папке confirmed_fronts
    samples = build_samples(root)

    split = make_splits(samples, val_size=0.15, test_size=0.15, seed=42, min_count_per_class=5)
    print("Classes:", len(split.class_to_idx))
    print("Train/Val/Test:", len(split.train), len(split.val), len(split.test))

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = DVMColorDataset(split.train, split.class_to_idx, transform=train_tfms)
    val_ds   = DVMColorDataset(split.val,   split.class_to_idx, transform=test_tfms)
    test_ds  = DVMColorDataset(split.test,  split.class_to_idx, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # быстрый sanity check
    xb, yb = next(iter(train_loader))
    print("Batch images:", xb.shape, xb.dtype)
    print("Batch labels:", yb.shape, yb.dtype, "min/max:", yb.min().item(), yb.max().item())

    print("OK")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # полезно для Windows
    main()
