import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo

class DepthwiseConv2d(nn.Module):
    """
    深度可分离卷积
    大幅减少参数量和计算量
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super(DepthwiseConv2d, self).__init__()

        # 深度卷积：每个输入通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

        # 点卷积：1x1卷积混合通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x