"""Checkpoint-compatible ResNet18 + U-Net with three attention gates.

The registered base_model retains the original unused classifier for strict loading.
"""
import torch
from torch import nn
from torchvision import models
import torchvision.transforms.functional as TF

class AttentionBlock(nn.Module):
    """
    注意力门控机制 (Attention Gate)：
    在跳跃连接(Skip Connection)处，利用深层语义特征(g)指导浅层空间特征(x)，
    抑制无关背景噪声，使模型更聚焦于目标区域（如微小的剩余油滴）。
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 尺寸保护机制：若深浅层特征尺寸不严格对齐，强制双线性插值对齐
        if g.shape[-2:] != x.shape[-2:]:
            g = TF.resize(g, size=x.shape[2:])
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionResUNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=False):
        super().__init__()
        # 引入预训练的 ResNet18 作为编码器主干，加速收敛并提取高级语义
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 编码器 (共5次空间下采样)
        self.encoder0 = nn.Sequential(self.base_model.conv1, self.base_model.bn1, self.base_model.relu)
        self.encoder1 = nn.Sequential(self.base_model.maxpool, self.base_model.layer1)
        self.encoder2 = self.base_model.layer2
        self.encoder3 = self.base_model.layer3
        self.encoder4 = self.base_model.layer4

        filters = [64, 64, 128, 256, 512]

        # 解码器 (引入转置卷积上采样与注意力机制)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[3] // 2)
        self.dc4 = DoubleConv(filters[4], filters[3])

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[2] // 2)
        self.dc3 = DoubleConv(filters[3], filters[2])

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[1] // 2)
        self.dc2 = DoubleConv(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.dc1 = DoubleConv(filters[0] + 64, filters[0])

        self.up0 = nn.ConvTranspose2d(filters[0], filters[0] // 2, kernel_size=2, stride=2)
        self.dc0 = DoubleConv(filters[0] // 2 + 3, filters[0] // 2)

        self.final_conv = nn.Conv2d(filters[0] // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x_input = x

        # 编码
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 解码4
        d4 = self.up4(e4)
        if d4.shape[-2:] != e3.shape[-2:]: d4 = TF.resize(d4, size=e3.shape[2:])
        e3 = self.att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.dc4(d4)

        # 解码3
        d3 = self.up3(d4)
        if d3.shape[-2:] != e2.shape[-2:]: d3 = TF.resize(d3, size=e2.shape[2:])
        e2 = self.att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.dc3(d3)

        # 解码2
        d2 = self.up2(d3)
        if d2.shape[-2:] != e1.shape[-2:]: d2 = TF.resize(d2, size=e1.shape[2:])
        e1 = self.att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.dc2(d2)

        # 解码1
        d1 = self.up1(d2)
        if d1.shape[-2:] != e0.shape[-2:]: d1 = TF.resize(d1, size=e0.shape[2:])
        d1 = torch.cat((e0, d1), dim=1)
        d1 = self.dc1(d1)

        # 解码0
        d0 = self.up0(d1)
        if d0.shape[-2:] != x_input.shape[-2:]: d0 = TF.resize(d0, size=x_input.shape[2:])
        d0 = torch.cat((x_input, d0), dim=1)
        d0 = self.dc0(d0)

        return self.final_conv(d0)
