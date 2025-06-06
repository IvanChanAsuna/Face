import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, Tuple, Optional
import json
from tqdm import tqdm
import warnings
import logging
from utils.AverageMeter import AverageMeter
import torchvision
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
class TrainerV3:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.writer = SummaryWriter(log_dir='/root/tf-logs/')


    def train(self, model, dataloader):
        try:
            params = [
                # 直接访问 base_model 而非 model.model.base_model
                {'params': model.base_model[10].parameters(), 'lr': self.args.v3_12_lr},
                {'params': model.base_model[11].parameters(), 'lr': self.args.v3_12_lr},
                {'params': model.base_model[12].parameters(), 'lr': self.args.v3_12_lr},
                # 直接访问 classifier
                {'params': model.classifier.parameters(), 'lr': self.args.v3_cl_lr}
            ]
            since = time.time()
            logger.info(model)
            logger.info(model.base_model)
            logger.info(model.classifier)
            optimizer = optim.AdamW(params, weight_decay=self.args.v3_weight_decay, betas=(0.9, 0.999))
            logger.info(f'learning rate_12: {optimizer.param_groups[0]["lr"]}, learning rate_cl: {optimizer.param_groups[3]["lr"]}')

            # 计算总训练步数，用于学习率调度
            total_steps = len(dataloader['train']) * self.args.warm_up_epochs  # 大致的一个训练step数

            # 创建带预热的余弦学习率调度器
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)


            logger.info("Start training...")
            epochs, best_epoch, best_valid = 0, 0, 0
            losses = []
            valid_losses = []
            valid_F1 = []
            lr = []
            # loop util earlystop
            while True:
                epochs += 1
                model.train()
                train_loss = 0.0
                # 梯度累积的剩余批次数
                left_epochs = self.args.update_epochs
                ids = []
                with tqdm(dataloader['train'],
                          desc=f'Epoch {epochs}',
                          bar_format='{l_bar}{bar:20}{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as td:
                    for batch_idx, batch_data in enumerate(td):
                        if left_epochs == self.args.update_epochs:
                            optimizer.zero_grad()  # 在训练1个batch之后停止梯度清0，当新的epoch来临时才清0
                        left_epochs -= 1  # 这么做相当于把batch_size扩大为（N-1）*batch_size，其中N为一个epoch中的batch数

                        # 将数据移到设备上
                        images, labels = batch_data
                        images = images.to(self.args.device)
                        labels = labels.to(self.args.device)

                        optimizer.zero_grad()
                        # forward
                        _, logits = model(images)
                        weights = torch.tensor(self.args.v3_class_weights).to(self.args.device)
                        criterion = nn.CrossEntropyLoss(weight=weights)

                        total_loss = criterion(logits, labels)


                        # backward
                        total_loss.backward()
                        train_loss += total_loss.item()
                        avg_train_loss = train_loss / (batch_idx + 1)  # 累积平均
                        # ---- 实时更新进度条信息 ----
                        td.set_postfix(
                            batch=f"{batch_idx + 1}/{len(dataloader['train'])}",  # 当前批次/总批次
                            loss=f"{total_loss.item():.4f}",  # 总损失
                            avg_loss=f"{train_loss / (batch_idx + 1):.4f}",  # 累积平均
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}",  # 学习率
                            accum_step=f"{self.args.update_epochs - left_epochs}/{self.args.update_epochs}"  # 梯度累积进度
                        )
                        lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                        # update parameters
                        if not left_epochs:
                            # update
                            optimizer.step()
                            scheduler.step()
                            left_epochs = self.args.update_epochs
                    # 处理最后一批次可能不满梯度累积数的情况
                    if not left_epochs:
                        # update
                        optimizer.step()
                train_loss = train_loss / len(dataloader['train'])
                logger.debug("TRAIN-(%s-%s) (%d/%d/%d)>> loss: %.4f" % (self.args.modelName, self.args.datasetName, \
                                                                                    epochs - best_epoch, epochs,                                                                  self.args.cur_time, train_loss))
                losses.append(train_loss)

                cur_valid = self.do_test(model, dataloader, mode="val", epoch=epochs)

                valid_losses.append(cur_valid['loss'])
                valid_F1.append(cur_valid)
                # save best model
                # 检查是否是最佳模型
                isBetter = False
                if self.args.KeyEval == 'f1':
                    isBetter = cur_valid['f1_score'] >= (best_valid + 1e-6)
                elif self.args.KeyEval == 'roc_auc':
                    isBetter = cur_valid['roc_auc'] >= (best_valid + 1e-6)
                else:  # 默认使用f1
                    isBetter = cur_valid['f1_score'] >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid['f1_score'], epochs
                    # save model
                    torch.save(model.cpu().state_dict(), self.args.model_param_save_path)
                    model.to(self.args.device)

                # early stop
                if epochs - best_epoch >= self.args.early_stop:  # 如果比best_epoch再过了early_stop轮之后还没有出现新的best_epoch，就停止训练
                    return
        finally:
            self.writer.close()

    def do_test(self, model, dataloader, mode="val", epoch=None):
        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []  # 新增：存储所有的预测概率
        sample_images = []  # 存储样本图像用于可视化
        sample_preds = []  # 存储样本预测结果
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        criterion = nn.CrossEntropyLoss()
        first = True
        with torch.no_grad():
            with tqdm(dataloader[mode]) as td:
                for batch_data in td:
                    images, labels = batch_data
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.long().to(self.device, non_blocking=True)


                    _, logits = model(images)
                    loss = criterion(logits, labels)

                    # 计算概率
                    probabilities = nn.functional.softmax(logits, dim=1)
                    positive_probs = probabilities[:, 1]  # 获取正类的概率

                    # 计算准确率
                    _, predicted = torch.max(logits.data, 1)
                    accuracy = (predicted == labels).float().mean()
                    if first:
                        sample_images.append(images.cpu())
                        sample_preds.extend(predicted.cpu().numpy())
                        first = False

                    # 更新指标
                    loss_meter.update(loss.item(), images.size(0))
                    acc_meter.update(accuracy.item(), images.size(0))

                    # 收集预测结果 - 移动到CPU
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(positive_probs.cpu().numpy())  # 保存概率

        if len(sample_images) > 0:
            # 将图像合并为网格
            img_grid = torchvision.utils.make_grid(
                torch.cat(sample_images[:8]),  # 最多显示8张图像
                nrow=4, normalize=True, scale_each=True
            )

            # 添加图像到TensorBoard
            self.writer.add_image(f'{mode}/predictions', img_grid, epoch)

            # 添加预测结果对比
            for i, (img, pred, label) in enumerate(zip(sample_images[0], sample_preds, all_labels[:len(sample_preds)])):
                self.writer.add_text(
                    f'{mode}/sample_{i}',
                    f'True: {label}, Pred: {pred}',
                    epoch
                )
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算各项指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'  # 修改为binary
        )

        try:
            # 确保长度一致后计算ROC AUC
            roc_auc = roc_auc_score(all_labels, all_probs)
        except Exception as e:
            logger.error(f"could not compute ROC AUC: {str(e)}")
            logger.error(f"Labels shape: {all_labels.shape}, Probs shape: {all_probs.shape}")
            roc_auc = 0.0

        # 计算混淆矩阵相关指标
        cm = confusion_matrix(all_labels, all_predictions)
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        logger.info(f'{mode}: f1:{f1:.4f} '
                    f'accuracy: {acc_meter.avg:.4f} '
                    f'loss: {loss_meter.avg:.4f} '
                    f'precision: {precision:.4f} '
                    f'recall: {recall:.4f} '
                    f'roc_auc: {roc_auc:.4f} '
                    f'specificity: {specificity:.4f} ')
        logger.info(f'pred: {all_predictions}')
        logger.info(f'labels: {all_labels}')

        return {
            'loss': loss_meter.avg,
            'accuracy': acc_meter.avg,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'specificity': specificity,
        }

    def count_samples(self, dataloader: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """
        统计各数据集中正负样本数量
        Args:
            dataloader: 包含'train', 'val', 'test'的字典
        Returns:
            counts: 字典格式统计结果，如 {'train': {'positive': 100, 'negative': 200}}
        """
        counts = {}
        for mode in ['train', 'val', 'test']:
            if mode not in dataloader:
                continue

            positive = 0
            negative = 0

            # 遍历数据集
            for batch_data in dataloader[mode]:
                _, labels = batch_data
                # 统计当前batch的正负样本
                positive += (labels == 1).sum().item()
                negative += (labels == 0).sum().item()

            counts[mode] = {
                'positive': positive,
                'negative': negative,
                'total': positive + negative,
                'positive_ratio': positive / (positive + negative) * 100
            }

        return counts
