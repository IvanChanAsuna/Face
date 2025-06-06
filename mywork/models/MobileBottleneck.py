import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo

class MobileBottleneck(nn.Module):
    """
    MobileNet瓶颈块
    使用inverted residual结构
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, expansion: int = 6):
        super(MobileBottleneck, self).__init__()

        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # 计算扩展通道数
        hidden_dim = in_channels * expansion

        layers = []

        # 扩展阶段（如果扩展因子不为1）
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # 深度卷积阶段
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            # 压缩阶段
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)