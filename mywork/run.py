import gc
import logging
import os
import datetime
import pandas as pd
from configs import Configs
import torch
import numpy as np
from typing import List, Tuple  # 添加这行导入
import random
from data.load_dataset import MMDataLoader
import os
import sys
import json
from models.FaceRecognitionModel import FaceRecognitionModel
from models.MobileNet_v3 import MobileNetV3
# 添加项目根目录到Python路径
from train.train import Trainer
from train.train_v3 import TrainerV3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f"logs/{current_time}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)

    formatter_file = logging.Formatter(
        '%(asctime)s - %(module)s - %(levelname)s - %(message)s',  # 添加 %(module)s
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    formatter_stream = logging.Formatter(
        '%(asctime)s - %(module)s - %(levelname)s - %(message)s',  # 添加 %(module)s
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_train(args):
    # if not os.path.exists(args.model_save_dir):
    #     os.makedirs(args.model_save_dir)
    os.makedirs(args.model_save_dir, exist_ok=True)

    args.model_param_save_path = os.path.join(args.model_save_dir, \
                                              f'{args.seed}.pth')
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')
    args.device = device

    dataloader = MMDataLoader(args)

    model = FaceRecognitionModel(args).to(device)
    trainer = Trainer(args)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer

    # def calculate_dataset_stats(dataloaders: dict) -> Tuple[List[float], List[float]]:
    #     """
    #     计算数据集各通道的均值和标准差
    #
    #     Args:
    #         dataloaders: 包含'train','val','test'模式的DataLoader字典
    #
    #     Returns:
    #         mean: 各通道均值 [R_mean, G_mean, B_mean]
    #         std: 各通道标准差 [R_std, G_std, B_std]
    #     """
    #     # 初始化统计变量
    #     mean = torch.zeros(3)
    #     std = torch.zeros(3)
    #     total_pixels = 0
    #
    #     # 遍历所有数据加载器
    #     for mode in ['train', 'val', 'test']:
    #         if mode not in dataloaders:
    #             continue
    #
    #         loader = dataloaders[mode]
    #         for images, _ in loader:
    #             # 转换图像为 [C, H*W*B] 格式
    #             b, c, h, w = images.shape
    #             images_flat = images.view(b, c, -1)  # [B, C, H*W]
    #
    #             # 累加统计量
    #             for i in range(3):
    #                 channel_data = images_flat[:, i, :]  # 当前通道所有像素
    #                 mean[i] += channel_data.sum()
    #                 std[i] += (channel_data ** 2).sum()
    #
    #             total_pixels += b * h * w
    #
    #     # 计算最终统计量
    #     mean /= total_pixels
    #     std = torch.sqrt(std / total_pixels - mean ** 2)
    #
    #     return mean.numpy().tolist(), std.numpy().tolist()
    #
    # # 训练前调用
    # train_mean, train_std = calculate_dataset_stats(dataloader)
    # args.normalize_mean = train_mean
    # args.normalize_std = train_std

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # logger.info(f'mean: {train_mean}, std: {train_std}')
    sample_counts = trainer.count_samples(dataloader)
    logger.info(f'Sample: {json.dumps(sample_counts, indent=2)}')
    trainer.train(model=model, dataloader=dataloader)
    assert os.path.exists(args.model_param_save_path)
    model.load_state_dict(torch.load(args.model_param_save_path))
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = trainer.do_test(model, dataloader, mode="val", epoch=0)
    else:
        results = trainer.do_test(model, dataloader, mode="test", epoch=0)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run(args):
    model_results = []
    os.makedirs(args.res_save_dir, exist_ok=True)
    save_path = os.path.join(args.res_save_dir,
                             f'{current_time}.csv')

    args.model_save_dir = os.path.join(args.model_save_dir, f'{current_time}')
    logger.info(args)
    # runnning
    test_results = run_train(args)  # 训练
    # restore results
    model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results

    # if not os.path.exists(args.res_save_dir):
    #     os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        # df = pd.DataFrame(columns=["Model"] + criterions)
        df = pd.DataFrame(columns=criterions)
    # save results
    # res = [args.modelName]

    for i, test_results in enumerate(model_results):
        res = []
        for c in criterions:
            res.append(round(test_results[c], 2))
        df.loc[len(df)] = res

    # df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' % (save_path))
    df = df.iloc[0:0]  # 保存后清0
    model_results = []


def run_train_v3(args):
    # if not os.path.exists(args.model_save_dir):
    #     os.makedirs(args.model_save_dir)
    os.makedirs(args.model_save_dir, exist_ok=True)

    args.model_param_save_path = os.path.join(args.model_save_dir, \
                                              f'{args.seed}_v3.pth')
    args.model_save = os.path.join(args.model_save_dir, f'all_{args.seed}_v3.pth')
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')
    args.device = device

    dataloader = MMDataLoader(args)

    model = MobileNetV3(args).to(device)
    trainer = TrainerV3(args)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
                # print(p)
        return answer

    # def calculate_dataset_stats(dataloaders: dict) -> Tuple[List[float], List[float]]:
    #     """
    #     计算数据集各通道的均值和标准差
    #
    #     Args:
    #         dataloaders: 包含'train','val','test'模式的DataLoader字典
    #
    #     Returns:
    #         mean: 各通道均值 [R_mean, G_mean, B_mean]
    #         std: 各通道标准差 [R_std, G_std, B_std]
    #     """
    #     # 初始化统计变量
    #     mean = torch.zeros(3)
    #     std = torch.zeros(3)
    #     total_pixels = 0
    #
    #     # 遍历所有数据加载器
    #     for mode in ['train', 'val', 'test']:
    #         if mode not in dataloaders:
    #             continue
    #
    #         loader = dataloaders[mode]
    #         for images, _ in loader:
    #             # 转换图像为 [C, H*W*B] 格式
    #             b, c, h, w = images.shape
    #             images_flat = images.view(b, c, -1)  # [B, C, H*W]
    #
    #             # 累加统计量
    #             for i in range(3):
    #                 channel_data = images_flat[:, i, :]  # 当前通道所有像素
    #                 mean[i] += channel_data.sum()
    #                 std[i] += (channel_data ** 2).sum()
    #
    #             total_pixels += b * h * w
    #
    #     # 计算最终统计量
    #     mean /= total_pixels
    #     std = torch.sqrt(std / total_pixels - mean ** 2)
    #
    #     return mean.numpy().tolist(), std.numpy().tolist()
    #
    # # 训练前调用
    # train_mean, train_std = calculate_dataset_stats(dataloader)
    # args.normalize_mean = train_mean
    # args.normalize_std = train_std

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # logger.info(f'mean: {train_mean}, std: {train_std}')
    sample_counts = trainer.count_samples(dataloader)
    logger.info(f'Sample: {json.dumps(sample_counts, indent=2)}')
    trainer.train(model=model, dataloader=dataloader)
    assert os.path.exists(args.model_param_save_path)
    model.load_state_dict(torch.load(args.model_param_save_path))
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = trainer.do_test(model, dataloader, mode="val", epoch=0)
    else:
        results = trainer.do_test(model, dataloader, mode="test", epoch=0)
    torch.save(model, args.model_save)
    del model
    torch.cuda.empty_cache()
    gc.collect()


    return results


def run_v3(args):
    model_results = []
    os.makedirs(args.res_save_dir, exist_ok=True)
    save_path = os.path.join(args.res_save_dir,
                             f'{current_time}_v3.csv')

    args.model_save_dir = os.path.join(args.model_save_dir, f'{current_time}')
    logger.info(args)
    # runnning
    test_results = run_train_v3(args)  # 训练
    # restore results
    model_results.append(test_results)

    criterions = list(model_results[0].keys())
    # load other results

    # if not os.path.exists(args.res_save_dir):
    #     os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        # df = pd.DataFrame(columns=["Model"] + criterions)
        df = pd.DataFrame(columns=criterions)
    # save results
    # res = [args.modelName]

    for i, test_results in enumerate(model_results):
        res = []
        for c in criterions:
            res.append(round(test_results[c], 2))
        df.loc[len(df)] = res

    # df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' % (save_path))
    df = df.iloc[0:0]  # 保存后清0
    model_results = []


if __name__ == '__main__':
    args = Configs()
    args = args.get_config()
    args.res_save_dir = os.path.join(args.res_save_dir, f'{current_time}')

    logger = set_log(args)
    logger.info(args)
    if args.train_v3:
        logger.info('========v3=========')
        run_v3(args)
    else:
        logger.info('======normal=======')
        run(args)

    # os.system("/usr/bin/shutdown")
