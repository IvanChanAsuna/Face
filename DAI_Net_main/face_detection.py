# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from DAI_Net_main.models.factory import build_net
from torchvision.utils import make_grid
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class FaceDetector:
    def __init__(self, model_path='DAI_Net_main/weights/DarkFaceZSDA.pth', score_threshold=0.7):
        self.use_cuda = torch.cuda.is_available()
        self.model_path = model_path

        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = False  # 修复拼写错误
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.USE_MULTI_SCALE = True  # 是否使用多尺度检测
        self.MY_SHRINK = 1  # 缩放参数
        self.score_threshold = score_threshold  # 置信度阈值

        # 加载模型
        self.net = self.load_models()

    @staticmethod
    def tensor_to_image(tensor):
        """将tensor转换为图片"""
        grid = make_grid(tensor)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return ndarr

    @staticmethod
    def to_chw_bgr(image):
        """
        Transpose image from HWC to CHW and from RGB to BGR.
        Args:
            image (np.array): an image with HWC and RGB layout.
        """
        # HWC to CHW
        if len(image.shape) == 3:
            image = np.swapaxes(image, 1, 2)
            image = np.swapaxes(image, 1, 0)
        # RGB to BGR
        image = image[[2, 1, 0], :, :]
        return image

    def detect_face(self, img, tmp_shrink):
        """单尺度人脸检测"""
        image = cv2.resize(img, None, None, fx=tmp_shrink,
                           fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)

        x = self.to_chw_bgr(image)
        x = x.astype('float32')
        x = x / 255.
        x = x[[2, 1, 0], :, :]

        x = Variable(torch.from_numpy(x).unsqueeze(0))

        if self.use_cuda:
            x = x.cuda()

        y = self.net.test_forward(x)[0]
        detections = y.data.cpu().numpy()
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        boxes = []
        scores = []
        for i in range(detections.shape[1]):
            j = 0
            while ((j < detections.shape[2]) and detections[0, i, j, 0] > 0.0):
                pt = (detections[0, i, j, 1:] * scale)
                score = detections[0, i, j, 0]
                boxes.append([pt[0], pt[1], pt[2], pt[3]])
                scores.append(score)
                j += 1

        det_conf = np.array(scores)
        boxes = np.array(boxes)

        if boxes.shape[0] == 0:
            return np.array([[0, 0, 0, 0, 0.001]])

        det_xmin = boxes[:, 0]
        det_ymin = boxes[:, 1]
        det_xmax = boxes[:, 2]
        det_ymax = boxes[:, 3]
        det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

        return det

    def flip_test(self, image, shrink):
        """翻转测试"""
        image_f = cv2.flip(image, 1)
        det_f = self.detect_face(image_f, shrink)

        det_t = np.zeros(det_f.shape)
        det_t[:, 0] = image.shape[1] - det_f[:, 2]
        det_t[:, 1] = det_f[:, 1]
        det_t[:, 2] = image.shape[1] - det_f[:, 0]
        det_t[:, 3] = det_f[:, 3]
        det_t[:, 4] = det_f[:, 4]
        return det_t

    def multi_scale_test(self, image, max_im_shrink):
        """多尺度测试"""
        # shrink detecting and shrink only detect big face
        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        det_s = self.detect_face(image, st)
        if max_im_shrink > 0.75:
            det_s = np.row_stack((det_s, self.detect_face(image, 0.75)))
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
        det_s = det_s[index, :]

        # enlarge one times
        bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        det_b = self.detect_face(image, bt)

        # enlarge small image x times for small face
        if max_im_shrink > 1.5:
            det_b = np.row_stack((det_b, self.detect_face(image, 1.5)))
        if max_im_shrink > 2:
            bt *= 2
            while bt < max_im_shrink:
                det_b = np.row_stack((det_b, self.detect_face(image, bt)))
                bt *= 2

            det_b = np.row_stack((det_b, self.detect_face(image, max_im_shrink)))

        # enlarge only detect small face
        if bt > 1:
            index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
            det_b = det_b[index, :]
        else:
            index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
            det_b = det_b[index, :]

        return det_s, det_b

    def multi_scale_test_pyramid(self, image, max_shrink):
        """金字塔多尺度测试"""
        det_b = self.detect_face(image, 0.25)
        index = np.where(
            np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
            > 30)[0]
        det_b = det_b[index, :]

        st = [1.25, 1.75, 2.25]
        for i in range(len(st)):
            if (st[i] <= max_shrink):
                det_temp = self.detect_face(image, st[i])
                # enlarge only detect small face
                if st[i] > 1:
                    index = np.where(
                        np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                    det_temp = det_temp[index, :]
                else:
                    index = np.where(
                        np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                                   det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                    det_temp = det_temp[index, :]
                det_b = np.row_stack((det_b, det_temp))
        return det_b

    @staticmethod
    def bbox_vote(det_):
        """边界框投票"""
        order_ = det_[:, 4].ravel().argsort()[::-1]
        det_ = det_[order_, :]
        dets_ = np.zeros((0, 5), dtype=np.float32)
        while det_.shape[0] > 0:
            # IOU
            area_ = (det_[:, 2] - det_[:, 0] + 1) * (det_[:, 3] - det_[:, 1] + 1)
            xx1 = np.maximum(det_[0, 0], det_[:, 0])
            yy1 = np.maximum(det_[0, 1], det_[:, 1])
            xx2 = np.minimum(det_[0, 2], det_[:, 2])
            yy2 = np.minimum(det_[0, 3], det_[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o_ = inter / (area_[0] + area_[:] - inter)

            # get needed merge det and delete these det
            merge_index_ = np.where(o_ >= 0.3)[0]
            det_accu_ = det_[merge_index_, :]
            det_ = np.delete(det_, merge_index_, 0)

            if merge_index_.shape[0] <= 1:
                continue
            det_accu_[:, 0:4] = det_accu_[:, 0:4] * np.tile(det_accu_[:, -1:], (1, 4))
            max_score_ = np.max(det_accu_[:, 4])
            det_accu_sum_ = np.zeros((1, 5))
            det_accu_sum_[:, 0:4] = np.sum(det_accu_[:, 0:4], axis=0) / np.sum(det_accu_[:, -1:])
            det_accu_sum_[:, 4] = max_score_
            try:
                dets_ = np.row_stack((dets_, det_accu_sum_))
            except:
                dets_ = det_accu_sum_

        dets_ = dets_[0:750, :]
        return dets_


    def draw_bbox_on_image(self, image, dets, score_threshold=0.5):
        """
        在图片上绘制检测框
        Args:
            image: PIL Image对象或numpy数组
            dets: 检测结果，格式为[xmin, ymin, xmax, ymax, score]
            score_threshold: 置信度阈值
        Returns:
            绘制了检测框的PIL Image
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 创建副本避免修改原图
        image = image.copy()

        # 创建绘制对象
        draw = ImageDraw.Draw(image)

        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # 绘制每个检测框
        for i in range(dets.shape[0]):
            score = dets[i][4]
            if score > score_threshold:
                xmin = int(dets[i][0])
                ymin = int(dets[i][1])
                xmax = int(dets[i][2])
                ymax = int(dets[i][3])

                # 绘制矩形框
                draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)

                # 绘制置信度标签
                label = f"Face: {score:.3f}"
                # 获取文本边界框
                bbox = draw.textbbox((xmin, ymin - 25), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 绘制背景矩形
                draw.rectangle([xmin, ymin - 25, xmin + text_width, ymin], fill="red")
                # 绘制文本
                draw.text((xmin, ymin - 25), label, fill="white", font=font)

        return image

    def load_models(self):
        """加载模型"""
        print('build network')
        net = build_net('test', num_classes=2, model='dark')
        net.eval()

        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 加载模型权重
        if self.use_cuda:
            net.load_state_dict(torch.load(self.model_path))
            net = net.cuda()
        else:
            net.load_state_dict(torch.load(self.model_path, map_location='cpu'))

        return net

    def detect_single_image(self, image, save_path='./result/'):
        """
        检测单张图片并可视化结果
        Args:
            image_path: 输入图片路径
            save_path: 保存路径
            score_threshold: 置信度阈值
        Returns:
            result_image: 检测结果图片
            dets: 检测框信息
        """
        score_threshold = self.score_threshold
        # image_path = 'path'
        # 加载图片
        if image.mode == 'L':
            image = image.convert('RGB')
        image_array = np.array(image)

        # 人脸检测
        max_im_shrink = (0x7fffffff / 200.0 / (image_array.shape[0] * image_array.shape[1])) ** 0.5
        max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

        with torch.no_grad():
            if self.USE_MULTI_SCALE:
                det0 = self.detect_face(image_array, self.MY_SHRINK)
                print("detect face finished")
                det1 = self.flip_test(image_array, self.MY_SHRINK)
                print("flip test finished")
                [det2, det3] = self.multi_scale_test(image_array, max_im_shrink)
                print("multi scale test finished")
                det4 = self.multi_scale_test_pyramid(image_array, max_im_shrink)
                print("multi scale test pyramid finished")
                det = np.row_stack((det0, det1, det2, det3, det4))
                dets = self.bbox_vote(det)
                print("bbox vote finished")
            else:
                dets = self.detect_face(image_array, self.MY_SHRINK)

        # 绘制检测框
        result_image = self.draw_bbox_on_image(image, dets, score_threshold=score_threshold)

        # 保存结果
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # # 保存可视化图片
        # base_name = Path(os.path.basename(image_path)).stem
        # # result_image.save(os.path.join(save_path, f'{base_name}_result.jpg'))
        #
        # with open(os.path.join(save_path, f'{base_name}.txt'), 'w') as fout:
        #     for i in range(dets.shape[0]):
        #         if dets[i][4] > score_threshold:
        #             xmin, ymin, xmax, ymax, score = dets[i]
        #             fout.write(f'{xmin} {ymin} {xmax} {ymax} {score}\n')
        # #
        # # print(f'检测到 {np.sum(dets[:, 4] > score_threshold)} 个人脸')
        # # print(f'结果已保存到: {os.path.join(save_path, f"{base_name}_result.jpg")}')

        return result_image

    def detect_batch_images(self, input_dir, save_path='./result/', score_threshold=0.5):
        """
        批量检测图片
        Args:
            input_dir: 输入图片目录
            save_path: 保存路径
            score_threshold: 置信度阈值
        """
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

        if not image_files:
            print(f"在目录 {input_dir} 中未找到图片文件")
            return

        print(f"找到 {len(image_files)} 张图片")

        for i, image_path in enumerate(image_files):
            print(f"处理第 {i + 1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
            try:
                self.detect_single_image(image_path, save_path, score_threshold)
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")
                continue

        print("批量处理完成！")


# 使用示例
if __name__ == "__main__":
    # 创建检测器实例
    detector = FaceDetector(model_path='./weights/DarkFaceZSDA.pth')

    # 检测单张图片
    try:
        result_img, detections = detector.detect_single_image(
            image_path='./test_image.jpg',
            save_path='./results/',
            score_threshold=0.5
        )
        print("检测完成！")
    except Exception as e:
        print(f"检测失败: {str(e)}")

    # 批量检测图片
    # detector.detect_batch_images(
    #     input_dir='./test_images/',
    #     save_path='./results/',
    #     score_threshold=0.5
    # )