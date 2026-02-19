# models_resnet18_scratch.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=True)
        out = self.conv2(out); out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int):
        super().__init__()
        self.in_ch = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, out_ch: int, blocks: int, stride: int):
        layers = []
        layers.append(block(self.in_ch, out_ch, stride=stride))
        self.in_ch = out_ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_scratch(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
