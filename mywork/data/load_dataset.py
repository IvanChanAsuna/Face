import logging
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import random
from typing import List, Tuple, Dict, Optional, Union
import glob
import json
import pickle
from collections import defaultdict
import albumentations as A
from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN  # 新增关键依赖
import hashlib
from torch.utils.data import DataLoader, Dataset

from mywork.configs import Configs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除现有处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# 添加处理器到记录器
logger.addHandler(console_handler)

class MMDataset(Dataset):
    def __init__(self,
                 args,
                 mode='train',
                 transform: Optional[Union[transforms.Compose, A.Compose]] = None):
        self.args = args
        self.transform = transform
        self.mode = mode
        self.self_images_path = args.self_images_path
        self.lfw_path = args.lfw_path
        self.force_realign = getattr(args, 'force_realign', False)

        self.positive_cache_dir = os.path.join(args.cache_root, 'positive_faces')
        self.negative_cache_dir = os.path.join(args.cache_root, 'negative_faces')
        os.makedirs(self.positive_cache_dir, exist_ok=True)
        os.makedirs(self.negative_cache_dir, exist_ok=True)
        self.mtcnn = MTCNN(
            image_size=args.img_size,
            margin=40,  # 保留足够背景
            device=args.device,
            keep_all=False,  # 检测多张人脸
            thresholds=[0.4, 0.55, 0.65],  # 统一阈值设置
            min_face_size=50,  # 增大最小人脸尺寸
            factor=0.55,  # 增加检测密度
            post_process=True,  # 启用后处理
            select_largest = True,
        )

        self.positive_augment_factor = args.positive_augment_factor
        self.negative_sample_num = args.negative_sample_num
        self.__init_data()

        # 根据模式选择数据集
        self.data = self.data_map[mode]
        self.labels = self.label_map[mode]
        del self.mtcnn

    def __init_data(self):
        logger.info(f'Loading dataset...')

        train_ratio = self.args.train_ratio
        val_ratio = self.args.val_ratio
        test_ratio = self.args.test_ratio
        positive_samples = self.__load_positive_samples()
        negative_samples = self.__load_negative_samples()

        X = positive_samples + negative_samples
        self.X = X
        y = [1] * len(positive_samples) + [0] * len(negative_samples)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            stratify=y,  # 分层抽样保持正负比例
            random_state=42  # 随机种子确保可复现
        )

        val_size = int(len(X_temp) * val_ratio / (val_ratio + test_ratio))
        test_size = len(X_temp) - val_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size,
            stratify=y_temp,
            random_state=self.args.seed
        )
        self.data_map = {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
        self.label_map = {
            'train': np.array(y_train),
            'val': np.array(y_val),
            'test': np.array(y_test)
        }
        logger.info(f'len(X_train) = {len(X_train)}')
        logger.info(f'len(X_val) = {len(X_val)}')
        logger.info(f'len(X_test) = {len(X_test)}')

    def __align_and_cache_image(self, img_path: str, is_positive: bool) -> Optional[str]:
        """人脸对齐并缓存图像，按正负样本分离存储"""
        file_hash = hashlib.md5(img_path.encode()).hexdigest()

        # 根据正负样本选择不同目录
        cache_dir = self.positive_cache_dir if is_positive else self.negative_cache_dir
        aligned_path = os.path.join(cache_dir, f"{file_hash}.jpg")

        # 检查缓存
        if not self.force_realign and os.path.exists(aligned_path):
            return aligned_path

        try:
            img = Image.open(img_path).convert('RGB')
            aligned_img = self.mtcnn(img, save_path=aligned_path)

            if aligned_img is None:
                logger.warning(f"未检测到人脸: {img_path}")
                return None

            return aligned_path
        except Exception as e:
            logger.error(f"对齐失败 {img_path}: {str(e)}")
            return None

    def __load_positive_samples(self) -> List[str]:
        if not os.path.exists(self.self_images_path):
            logger.error(f'{self.self_images_path} does not exist')
            raise FileNotFoundError(f"用户图片路径不存在: {self.self_images_path}")
        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

        # get all files
        positive_files = []
        for ext in image_extensions:
            pattern = os.path.join(self.self_images_path, ext)
            positive_files.extend(glob.glob(pattern))

        if len(positive_files) == 0:
            logger.error(f'{self.self_images_path} does not contain any positive images')
            raise ValueError(f"在 {self.self_images_path} 中未找到图像文件")

        logger.info(f'Find {len(positive_files)} positive images')
        aligned_files = []
        success_count = 0

        for img_path in positive_files:
            # 传递 is_positive=True 参数
            cached_path = self.__align_and_cache_image(img_path, is_positive=True)
            if cached_path:
                aligned_files.append(cached_path)
                success_count += 1

        logger.info(f"正样本对齐完成: {success_count}/{len(positive_files)} 成功")
        return aligned_files

    def __load_negative_samples(self) -> List[str]:
        """
        加载负样本（LFW数据集）

        Returns:
            负样本文件路径列表
        """
        if not os.path.exists(self.lfw_path):
            logger.error(f'{self.lfw_path} does not exist')
            raise FileExistsError(f'{self.lfw_path} does not exist')

        # 遍历LFW目录结构
        negative_files = []
        additional_samples = []
        for person_dir in os.listdir(self.lfw_path):
            if 'chinese' in person_dir:
                person_images = []
                person_path = os.path.join(self.lfw_path, 'chinese')
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    pattern = os.path.join(person_path, ext)
                    person_images.extend(glob.glob(pattern))
                    additional_samples.extend(person_images)
                continue
            person_path = os.path.join(self.lfw_path, person_dir)
            if os.path.isdir(person_path):
                # 获取该人物的所有图片
                person_images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    pattern = os.path.join(person_path, ext)
                    person_images.extend(glob.glob(pattern))

                # 随机选择部分图片作为负样本
                if len(person_images) > 0:
                    # 每个人最多选择3张照片
                    max_per_person = min(self.args.max_per_person, len(person_images))
                    selected = random.sample(person_images, max_per_person)
                    negative_files.extend(selected)

        # 如果负样本太多，随机采样
        if len(negative_files) > self.negative_sample_num - 29:
            negative_files = random.sample(negative_files, self.negative_sample_num - 29)
        additional_samples = set(additional_samples)
        logger.info(f'additional {len(additional_samples)} negative images')
        negative_files.extend(additional_samples)


        logger.info(f"加载 {len(negative_files)} 张LFW负样本")

        aligned_files = []
        success_count = 0

        for img_path in negative_files:
            # 传递 is_positive=False 参数
            cached_path = self.__align_and_cache_image(img_path, is_positive=False)
            if cached_path:
                aligned_files.append(cached_path)
                success_count += 1

        logger.info(f"负样本对齐完成: {success_count}/{len(negative_files)} 成功")
        return aligned_files


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        try:
            # 使用PIL加载图像
            image = Image.open(img_path).convert('RGB')

            # 应用预处理变换
            if self.transform is not None:
                # 处理不同类型的transform
                if isinstance(self.transform, A.Compose):
                    # Albumentations 变换
                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    image = transformed['image']
                    image = torch.from_numpy(image).permute(2, 0, 1).float()
                else:
                    # torchvision 变换
                    image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # 返回空图像作为占位符
            return torch.zeros(3, self.args.image_size, self.args.image_size), label

    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重，用于处理类别不平衡

        Returns:
            类别权重张量
        """
        positive_count = np.sum(self.labels == 1)
        negative_count = np.sum(self.labels == 0)
        total_count = len(self.labels)

        # 计算权重：样本数少的类别权重大
        weight_positive = total_count / (2 * positive_count)
        weight_negative = total_count / (2 * negative_count)

        return torch.tensor([weight_negative, weight_positive], dtype=torch.float32).to(self.args.device)

    def get_sampler(self) -> WeightedRandomSampler:
        """
        获取加权随机采样器

        Returns:
            加权随机采样器
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.labels]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

def get_train_transform(args):
    """训练集数据增强管道"""
    return A.Compose([
        # 1. 先执行尺寸无关的增强
        A.OneOf([
            A.HorizontalFlip(p=args.horizontal_flip_prob),
            A.VerticalFlip(p=0.1)  # 新增垂直翻转
        ], p=0.8),

        # 2. 执行几何变换（统一处理）
        A.ShiftScaleRotate(
            shift_limit=args.shift_limit,
            scale_limit=args.scale_limit,
            rotate_limit=args.rotate_limit,
            p=args.shift_scale_rotate_prob,
            border_mode=cv2.BORDER_REFLECT  # 反射填充避免黑边
        ),

        # 3. 统一尺寸
        A.Resize(height=args.img_size, width=args.img_size),

        # 4. 执行遮挡（在固定尺寸后）
        A.CoarseDropout(
            max_holes=args.cutout_num_holes,
            max_height=int(0.1 * args.img_size),  # 相对尺寸
            max_width=int(0.1 * args.img_size),
            fill_value=args.cutout_fill_value,
            p=args.cutout_prob
        ),

        # 5. 颜色变换
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=args.brightness_limit,
                contrast_limit=args.contrast_limit
            ),
            A.HueSaturationValue(
                hue_shift_limit=args.hue_shift_limit,
                sat_shift_limit=args.sat_shift_limit,
                val_shift_limit=args.val_shift_limit
            )
        ], p=args.hue_saturation_prob),

        # 6. 噪声增强（新增）
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # 7. 标准化
        A.Normalize(
            mean=args.normalize_mean,
            std=args.normalize_std,
            max_pixel_value=args.max_pixel_value
        )
    ])


def get_val_transform(args):
    """验证集数据转换管道"""
    return A.Compose([
        # 最后执行固定尺寸调整（确保输出尺寸一致）
        A.Resize(height=int(args.img_size), width=int(args.img_size)),
        A.Normalize(
            mean=args.normalize_mean,
            std=args.normalize_std,
            max_pixel_value=args.max_pixel_value
        )
    ])

def MMDataLoader(args):
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)
    test_transform = get_val_transform(args)
    datasets = {
        'train': MMDataset(args, mode='train', transform=train_transform),
        'val': MMDataset(args, mode='val', transform=val_transform),
        'test': MMDataset(args, mode='test', transform=test_transform),
    }

    logger.info(f'Loaded {len(datasets)} datasets')

    train_sampler = datasets['train'].get_sampler()
    dataLoader = {
        'train': DataLoader(datasets['train'],
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler=train_sampler,
                            pin_memory=True,
                            drop_last=True),  # 仅训练集需要shuffle
        'val': DataLoader(datasets['val'],
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=False,
                         drop_last=False),
        'test': DataLoader(datasets['test'],
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False,
                          drop_last=False)
    }
    return dataLoader

if __name__ == '__main__':
    args = Configs().get_config()
    test = MMDataLoader(args)
    print(test)