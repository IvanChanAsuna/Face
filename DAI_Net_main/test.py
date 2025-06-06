# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.factory import build_net
from torchvision.utils import make_grid
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benckmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def tensor_to_image(tensor):
    grid = make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def detect_face(img, tmp_shrink):
    image = cv2.resize(img, None, None, fx=tmp_shrink,
                       fy=tmp_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x = x / 255.
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))

    if use_cuda:
        x = x.cuda()

    y = net.test_forward(x)[0]
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

    det_xmin = boxes[:, 0]  # / tmp_shrink
    det_ymin = boxes[:, 1]  # / tmp_shrink
    det_xmax = boxes[:, 2]  # / tmp_shrink
    det_ymax = boxes[:, 3]  # / tmp_shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    return det


def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b, detect_face(image, 1.5)))
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:  # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def multi_scale_test_pyramid(image, max_shrink):
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
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


def bbox_vote(det_):
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


def draw_bbox_on_image(image, dets, score_threshold=0.5):
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
            draw.rectangle([xmin, ymin - 25, xmin + text_width, ymin], fill=(255, 0, 0, 128))
            # 绘制文本
            draw.text((xmin, ymin - 25), label, fill="white", font=font)

    return image


def load_models():
    print('build network')
    net = build_net('test', num_classes=2, model='dark')
    net.eval()
    # 请修改为你的模型权重路径
    net.load_state_dict(torch.load('E:\GXU\cv\DAI-Net-main\weights\DarkFaceZSDA.pth'))

    if use_cuda:
        net = net.cuda()

    return net


def detect_single_image(image_path, save_path='./result/', score_threshold=0.5):
    """
    检测单张图片并可视化结果
    Args:
        image_path: 输入图片路径
        save_path: 保存路径
        score_threshold: 置信度阈值
    """
    # 加载图片
    image = Image.open(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')
    image_array = np.array(image)

    # 人脸检测
    max_im_shrink = (0x7fffffff / 200.0 / (image_array.shape[0] * image_array.shape[1])) ** 0.5
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink

    with torch.no_grad():
        print(USE_MULTI_SCALE)
        if USE_MULTI_SCALE:
            det0 = detect_face(image_array, MY_SHRINK)
            det1 = flip_test(image_array, MY_SHRINK)
            [det2, det3] = multi_scale_test(image_array, max_im_shrink)
            det4 = multi_scale_test_pyramid(image_array, max_im_shrink)
            det = np.row_stack((det0, det1, det2, det3, det4))
            dets = bbox_vote(det)
        else:
            dets = detect_face(image_array, MY_SHRINK)

    # 绘制检测框
    result_image = draw_bbox_on_image(image, dets, score_threshold)

    # 保存结果
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存可视化图片
    base_name = Path(os.path.basename(image_path)).stem
    result_image.save(os.path.join(save_path, f'{base_name}_result.jpg'))

    # 同时保存检测结果文本文件
    with open(os.path.join(save_path, f'{base_name}.txt'), 'w') as fout:
        for i in range(dets.shape[0]):
            if dets[i][4] > score_threshold:
                xmin, ymin, xmax, ymax, score = dets[i]
                fout.write(f'{xmin} {ymin} {xmax} {ymax} {score}\n')

    print(f'检测到 {np.sum(dets[:, 4] > score_threshold)} 个人脸')
    print(f'结果已保存到: {os.path.join(save_path, f"{base_name}_result.jpg")}')

    return result_image, dets


if __name__ == '__main__':
    # 参数设置
    USE_MULTI_SCALE = True  # 是否使用多尺度检测
    MY_SHRINK = 1  # 缩放参数
    score_threshold = 0.7  # 置信度阈值
    muti = False

    # 路径设置
    save_path = './result/'

    # 加载模型
    net = load_models()

    # 方法1: 检测单张图片
    # 请修改为你的图片路径
    single_image_path = 'E:/GXU/cv/DAI-Net-main/dataset/1.jpeg'

    if os.path.exists(single_image_path):
        result_img, detections = detect_single_image(single_image_path, save_path, score_threshold)
        # 显示结果（如果在支持显示的环境中）
        result_img.show()

    if muti:
        # 方法2: 批量处理多张图片
        def load_images():
            # 请修改为你的测试数据目录
            imglist = glob.glob('E:/GXU/cv/DAI-Net-main/dataset/track1.2_test_sample/*.png')
            return imglist


        img_list = load_images()

        if img_list:
            print(f'开始处理 {len(img_list)} 张图片...')
            for idx, img_path in enumerate(img_list):
                print(f'处理第 {idx + 1}/{len(img_list)} 张图片: {os.path.basename(img_path)}')
                try:
                    detect_single_image(img_path, save_path, score_threshold)
                except Exception as e:
                    print(f'处理图片 {img_path} 时出错: {e}')
            print('批量处理完成！')
        else:
            print('未找到图片文件，请检查路径设置')