import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo


class ArcMarginProduct(nn.Module):
    """
    ArcFace损失的角度边距产品
    提高类间分离度
    """

    def __init__(self, in_features: int, out_features: int,
                 scale: float = 64.0, margin: float = 0.50, easy_margin: bool = False):
        """
        初始化ArcMarginProduct

        Args:
            in_features: 输入特征维度
            out_features: 输出类别数
            scale: 缩放因子
            margin: 角度边距
            easy_margin: 是否使用简单边距
        """
        super(ArcMarginProduct, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # 权重参数
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算的数值
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input: 输入特征 (B, in_features)
            label: 标签 (B,)

        Returns:
            输出logits (B, out_features)
        """
        # 归一化特征和权重
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # 计算cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 转换标签为one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # 应用边距
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output