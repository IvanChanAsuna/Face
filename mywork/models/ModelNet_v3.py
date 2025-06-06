import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.cuda.amp as amp


class MobileNetV3(nn.Module):
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
        super(MobileNetV3, self).__init__()
        self.args = args
        self.base_model = self._init_model()
        self.classifier = self._init_classifier()


    def _init_model(self):
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        # 冻结除最后三层外的参数
        for idx, param in enumerate(model.features.parameters()):
            if idx < 10:  # 冻结前10层
                param.requires_grad = False
        return model.features  # 仅返回特征提取器[1,7](@ref)

    def _init_classifier(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        features = self.base_model(x)  # 特征图 [B, 576, H, W]
        logits = self.classifier(features)  # 分类结果 [B, 2]
        return features, logits