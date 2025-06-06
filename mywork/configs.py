from mywork.utils.functions import Storage
import os
import torch
class Configs:
    def __init__(self):

        args = self.__mainParams()


        self.args = Storage(dict(
           **args
        ))

    def __mainParams(self):
        tmp = {
            # =============数据路径配置=============
            "self_images_path": "data/user_photos/",  # 用户自拍照片路径
            "lfw_path": "data/lfw/",  # LFW负样本数据集路径
            "output_path": "./output/",  # 输出路径
            "model_save_path": "./checkpoints/",  # 模型保存路径
            "log_path": "./logs/",  # 日志保存路径
            'train_v3': True,

            # =============图像预处理配置=============
            "img_size": 112,  # 输入图像尺寸
            "resize_width": 250,  # 调整宽度
            "resize_height": 250,  # 调整高度
            "img_mean": [0.5, 0.5, 0.5],  # 图像均值归一化
            "img_std": [0.5, 0.5, 0.5],  # 图像标准差归一化
            "scale": 1.0 / 255.0,  # 像素值缩放
            'margin': 20,
            'cache_root': "./dataset_cache" ,
            'force_realign': True,

            # =============数据增强参数=============
            # 几何变换参数
            "horizontal_flip_prob": 0.5,
            "rotate_limit": 15,
            "rotate_prob": 0.5,
            "random_scale_limit": 0.1,
            "random_scale_prob": 0.3,
            "shift_limit": 0.05,
            "scale_limit": 0.1,
            "shift_scale_rotate_prob": 0.5,

            # 颜色变换参数
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "brightness_contrast_prob": 0.5,
            "hue_shift_limit": 10,
            "sat_shift_limit": 20,
            "val_shift_limit": 10,
            "hue_saturation_prob": 0.5,

            # 遮挡增强参数
            "cutout_num_holes": 1,
            "cutout_max_h": 32,
            "cutout_max_w": 32,
            "cutout_fill_value": 0,
            "cutout_prob": 0.3,

            # 标准化参数
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "max_pixel_value": 255.0,

            # =============训练参数配置=============
            "batch_size": 64,  # 批次大小
            "learning_rate":5e-3,  # 初始学习率
            "num_epochs": 100,  # 训练轮数
            "weight_decay": 1e-4,  # 权重衰减
            "momentum": 0.9,  # SGD动量
            "warm_up_epochs": 20,  # 学习率预热轮数
            'update_epochs': 2,
            'v3_12_lr': 3e-4,
            'v3_cl_lr': 1e-3,
            'v3_weight_decay': 1e-5,
            'v3_class_weights': [1.0, 3.0],

            # =============模型架构配置=============
            "embedding_dim": 128,  # 特征嵌入维度
            "dropout2d_rate": 0.2,  # Dropout概率
            'post_se_dropout': 0.2,
            'feature_dropout': 0.2,
            'global_pool_dropout': 0.5,
            "backbone": "mobilefacenet",  # 主干网络类型
            "pretrained": True,  # 是否使用预训练权重
            'arc_scale': 64.0,
            'arc_margin': 0.5,
            'use_arcface': True,

            # =============损失函数配置=============
            "triplet_margin": 0.3,  # Triplet损失边距
            "triplet_weight": 1.0,  # Triplet损失权重
            "ce_weight": 1.0,  # 交叉熵损失权重
            "center_loss_weight": 0.003,  # Center Loss权重
            "center_loss_alpha": 0.5,  # Center Loss学习率

            # =============训练策略配置=============
            "scheduler_type": "cosine",  # 学习率调度器类型: step, cosine, exponential
            "scheduler_step_size": 30,  # StepLR步长
            "scheduler_gamma": 0.1,  # 学习率衰减因子
            "early_stop": 8,  # 早停耐心值
            "save_best_only": True,  # 是否只保存最佳模型

            # =============数据集配置=============
            "positive_augment_factor": 20,  # 正样本增强倍数
            # todo 正负样本比例
            "negative_sample_num": 2000,  # 负样本数量
            'max_per_person': 4,
            'max_val_negative_pairs':1000,
            "train_ratio": 0.7,  # 训练集比例
            "val_ratio": 0.15,  # 验证集比例
            "test_ratio": 0.15,  # 测试集比例
            'use_triplet': True,

            # =============硬件配置=============
            "device": "cuda",  # 设备
            "num_workers": 12,  # 数据加载线程数
            "pin_memory": True,  # 是否固定内存
            "mixed_precision": True,  # 是否使用混合精度训练

            # =============推理配置=============
            "similarity_threshold": 0.6,  # 相似度阈值
            "confidence_threshold": 0.9,  # 置信度阈值

            # =============人脸检测配置=============
            "min_face_size": 40,  # 最小人脸尺寸
            "face_detector": "mtcnn",  # 人脸检测器类型: mtcnn, retinaface
            "detection_device": "cuda",
            "keep_all": False,  # 是否保留所有检测到的人脸
            "min_face_confidence": 0.9,  # 人脸检测最小置信度

            # =============日志和可视化配置=============
            "log_interval": 10,  # 日志打印间隔
            "save_interval": 5,  # 模型保存间隔
            "tensorboard_log": True,  # 是否使用TensorBoard
            "plot_samples": True,  # 是否绘制训练样本

            # =============验证和测试配置=============
            "eval_interval": 1,  # 验证间隔
            "test_batch_size": 64,  # 测试批次大小
            "top_k": [1, 5],  # Top-K准确率
            'KeyEval': 'f1',

            # =============数据过滤配置=============
            "filter_min_face": True,  # 是否过滤小人脸
            "face_size_threshold": 32,  # 人脸尺寸阈值

            # =============模型量化配置=============
            "quantization": False,  # 是否进行模型量化
            "quantization_backend": "fbgemm",  # 量化后端

            # =============其他配置=============
            "seed": 520,  # 随机种子
            "deterministic": True,  # 是否使用确定性算法
            "benchmark": True,  # 是否启用cudnn benchmark
            'model_save_dir': 'results/models',
            'res_save_dir': 'results/data_res',
            'gpu_ids': 0,
        }
        return tmp



    def get_config(self):
        return self.args

# if __name__ == '__main__':
#     test = Configs()
#     args = test.get_config()
#     args.seeds = 5201314
#     print(args.batch_size)
