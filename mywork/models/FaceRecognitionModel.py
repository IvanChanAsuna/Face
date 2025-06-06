import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
import torch.utils.model_zoo as model_zoo
from mywork.models.MobileFaceNet import MobileFaceNet
from mywork.models.ArcMarginProduct import ArcMarginProduct

class FaceRecognitionModel(nn.Module):
    """
    完整的人脸识别模型
    结合特征提取和分类头
    """

    def __init__(self, args):
        """
        初始化人脸识别模型

        Args:
            config: 配置对象
        """
        super(FaceRecognitionModel, self).__init__()

        self.args = args
        self.num_classes = 2  # 二分类：用户本人 vs 其他人
        self.dropout = nn.Dropout(p=args.feature_dropout)  # 使用配置的dropout率[6,7](@ref)

        # 特征提取网络
        self.backbone = MobileFaceNet(
            embedding_dim=args.embedding_dim,
            args=args,
        )

        # 分类头
        self.classifier = nn.Linear(args.embedding_dim, self.num_classes)

        # ArcFace头（可选）
        self.use_arcface = hasattr(args, 'use_arcface') and args.use_arcface
        if self.use_arcface:
            self.arcface = ArcMarginProduct(
                in_features=args.embedding_dim,
                out_features=self.num_classes,
                scale=args.arc_scale,
                margin=args.arc_margin,
            )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像
            labels: 标签（训练时使用）

        Returns:
            输出logits
        """
        # 提取特征
        features = self.backbone(x)
        features = self.dropout(features)

        if self.use_arcface and labels is not None and self.training:
            # 训练时使用ArcFace
            logits = self.arcface(features, labels)
        else:
            # 推理时或不使用ArcFace时
            logits = self.classifier(features)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量

        Args:
            x: 输入图像

        Returns:
            特征向量
        """
        return self.backbone.extract_features(x)