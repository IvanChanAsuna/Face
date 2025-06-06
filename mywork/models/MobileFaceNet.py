import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo
from mywork.models.DepthwiseConv2d import DepthwiseConv2d
from mywork.models.MobileBottleneck import MobileBottleneck
from mywork.models.SEBlock import SEBlock
class MobileFaceNet(nn.Module):
    """
    轻量化人脸识别网络
    基于MobileNetV2和人脸识别优化设计
    """

    def __init__(self, embedding_dim: int = 128, args: dict = None,
                 width_multiplier: float = 1.0):
        """
        初始化MobileFaceNet

        Args:
            embedding_dim: 特征嵌入维度
            dropout_rate: Dropout概率
            width_multiplier: 宽度乘数，控制网络大小
        """
        super(MobileFaceNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.args = args
        # 计算通道数
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # 初始卷积层
        input_channel = _make_divisible(32 * width_multiplier)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )

        # 深度可分离卷积层
        self.conv2_dw = nn.Sequential(
            DepthwiseConv2d(input_channel, input_channel, 3, 1, 1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )

        # Bottleneck层配置
        # (expansion, out_channels, num_blocks, stride)
        bottleneck_configs = [
            (1, 64, 5, 2),  # 56x56
            (6, 128, 1, 2),  # 28x28
            (6, 128, 6, 1),  # 28x28
            (6, 128, 1, 2),  # 14x14
            (6, 128, 2, 1),  # 14x14
        ]

        # 构建Bottleneck层
        self.bottlenecks = nn.ModuleList()
        in_channels = input_channel

        for expansion, out_channels, num_blocks, stride in bottleneck_configs:
            out_channels = _make_divisible(out_channels * width_multiplier)

            for i in range(num_blocks):
                if i == 0:
                    self.bottlenecks.append(
                        MobileBottleneck(in_channels, out_channels, 3, stride, expansion)
                    )
                else:
                    self.bottlenecks.append(
                        MobileBottleneck(in_channels, out_channels, 3, 1, expansion)
                    )
                in_channels = out_channels

        # 最后的卷积层
        last_channel = _make_divisible(512 * width_multiplier)
        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )

        # SE注意力机制
        self.se = SEBlock(last_channel)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout层
        self.inter_block_dropout = nn.Dropout2d(args.dropout2d_rate)
        self.post_se_dropout = nn.Dropout(args.post_se_dropout)
        self.dropout = nn.Dropout(args.global_dropout)

        # 全连接层
        self.fc = nn.Linear(last_channel, embedding_dim, bias=False)

        # BatchNorm for embedding
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像张量 (B, 3, H, W)

        Returns:
            特征嵌入 (B, embedding_dim)
        """
        # 初始卷积
        x = self.conv1(x)  # (B, 32, 56, 56)

        # 深度可分离卷积
        x = self.conv2_dw(x)  # (B, 32, 56, 56)

        # Bottleneck层
        for i, bottleneck in enumerate(self.bottlenecks):
            x = bottleneck(x)
            if i % 2 == 0:  # 每隔两个块添加
                x = self.inter_block_dropout(x)  # [3](@ref)

        # 最后的卷积
        x = self.conv_last(x)  # (B, 512, 7, 7)

        # SE注意力
        x = self.se(x)
        x = self.post_se_dropout(x)

        # 全局平均池化
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)

        # Dropout
        x = self.dropout(x)

        # 全连接层
        x = self.fc(x)  # (B, embedding_dim)

        # BatchNorm
        x = self.bn_embedding(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征（用于推理）

        Args:
            x: 输入图像张量

        Returns:
            L2归一化的特征向量
        """
        features = self.forward(x)
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        return features