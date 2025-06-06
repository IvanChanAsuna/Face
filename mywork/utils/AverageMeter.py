import os
import torch
import numpy as np
import random
import logging
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import json
import pickle

class AverageMeter:
    """
    平均值计算器，用于统计训练过程中的指标
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计值"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新统计值

        Args:
            val: 新的值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count