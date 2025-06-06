import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation块
    通过全局信息重新校准通道权重
    """

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze
        y = self.squeeze(x).view(b, c)

        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)