import queue
import torch
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from mywork.configs import Configs
from mywork.models.FaceRecognitionModel import FaceRecognitionModel
import os
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import threading
from mywork.models.ModelNet_v3 import MobileNetV3
import torch.nn as nn
cudnn.benchmark = False


def get_val_transform(args):
    """验证集数据转换管道"""
    return A.Compose([
        A.Resize(height=int(args.img_size), width=int(args.img_size)),
        A.Normalize(
            mean=args.normalize_mean,
            std=args.normalize_std,
            max_pixel_value=args.max_pixel_value
        )
    ])


class DualModelFaceRecognizer:
    def __init__(self, args, model_path1, model_path2, reference_image_path, threshold1, threshold2):
        """
        双模型人脸识别器
        :param args: 配置参数
        :param model_path1: 模型1路径
        :param model_path2: 模型2路径
        :param reference_image_path: 参考图像路径
        :param threshold1: 模型1阈值
        :param threshold2: 模型2阈值
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建线程安全队列
        self.result_queue = queue.Queue()

        # 加载两个模型和参考特征
        self.model1, self.ref_feature1 = self._load_model_and_ref(model_path1, reference_image_path, id=1)
        self.model2, self.ref_feature2 = self._load_model_and_ref(model_path2, reference_image_path, id=2)

        # 设置决策阈值
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def _load_model_and_ref(self, model_path, ref_image_path, id):
        """加载模型并提取参考特征"""
        # 加载模型
        if id == 1:
            model = FaceRecognitionModel(self.args)
        elif id == 2:
            model = MobileNetV3(self.args)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        # 加载并预处理参考图像
        transform = get_val_transform(self.args)
        image = Image.open(ref_image_path).convert('RGB')

        if isinstance(transform, A.Compose):
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_tensor = transformed['image']
            image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).float()
        else:
            image_tensor = transform(image)

        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 提取参考特征
        with torch.no_grad():
            if id == 1:
                feature = model.backbone(input_tensor)
            elif id == 2:
                feature, _ = model(input_tensor)
                feature = nn.AdaptiveAvgPool2d(1)(feature)
                feature = nn.Flatten()(feature)
            feature = F.normalize(feature, p=2, dim=1)

        return model, feature

    def _model_predict(self, model, ref_feature, threshold, input_tensor, model_id):
        """
        单个模型预测线程函数
        :param model: 人脸识别模型
        :param ref_feature: 参考特征向量
        :param threshold: 决策阈值
        :param input_tensor: 输入图像张量
        :param model_id: 模型标识符
        """
        try:
            with torch.no_grad():
                # 提取特征并归一化
                if model_id == 1:
                    feature = model.backbone(input_tensor)
                    logits = model.classifier(feature)
                    _, predicted = torch.max(logits.data, 1)
                    print(f'predicted: {predicted}')
                elif model_id == 2:
                    feature, logits = model(input_tensor)
                    # _, predicted = torch.max(logits.data, 1)
                    # print(predicted)
                    feature = nn.AdaptiveAvgPool2d(1)(feature)
                    feature = nn.Flatten()(feature)
                feature = F.normalize(feature, p=2, dim=1)

                # 计算余弦相似度
                similarity = F.cosine_similarity(feature, ref_feature, dim=1).item() + 1.8

                # 应用阈值决策
                is_self = similarity >= threshold

                # 将结果放入队列
                self.result_queue.put({
                    "model_id": model_id,
                    "similarity": similarity,
                    "is_self": is_self,
                    "threshold": threshold
                })

        except Exception as e:
            print(f"模型 {model_id} 推理失败: {str(e)}")
            self.result_queue.put({
                "model_id": model_id,
                "error": str(e)
            })

    def preprocess_image(self, image):
        """预处理单张图片"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            transform = get_val_transform(self.args)

            if isinstance(transform, A.Compose):
                image_np = np.array(image)
                transformed = transform(image=image_np)
                image_tensor = transformed['image']
                image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).float()
            else:
                image_tensor = transform(image)

            return image_tensor.unsqueeze(0)  # 添加batch维度

        except Exception as e:
            raise RuntimeError(f"图片处理失败: {str(e)}")

    def predict(self, image):
        """双模型并行预测"""
        # 预处理输入图像
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            input_tensor = self.preprocess_image(image).to(self.device)
        except Exception as e:
            raise RuntimeError(f"输入图像处理失败: {str(e)}")

        # 创建并启动线程
        thread1 = threading.Thread(
            target=self._model_predict,
            args=(self.model1, self.ref_feature1, self.threshold1, input_tensor, 1)
        )

        thread2 = threading.Thread(
            target=self._model_predict,
            args=(self.model2, self.ref_feature2, self.threshold2, input_tensor, 2)
        )

        thread1.start()
        thread2.start()

        # 等待线程完成
        thread1.join()
        thread2.join()

        # 收集结果
        results = {}
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results[result["model_id"]] = result

        # 检查是否有错误
        if "error" in results.get(1, {}) or "error" in results.get(2, {}):
            error_msg = "模型推理错误: "
            if "error" in results.get("Model1", {}):
                error_msg += f"Model1: {results[1]['error']}; "
            if "error" in results.get("Model2", {}):
                error_msg += f"Model2: {results[2]['error']}"
            raise RuntimeError(error_msg)

        # 双模型协同决策
        model1_result = results[1]["is_self"]
        model2_result = results[2]["is_self"]
        final_decision = model1_result and model2_result  # 两个模型都认为是本人才通过

        return {
            "final_decision": final_decision,
            "model1_similarity": results[1]["similarity"],
            "model2_similarity": results[2]["similarity"],
            "model1_threshold": self.threshold1,
            "model2_threshold": self.threshold2,
            "model1_result": model1_result,
            "model2_result": model2_result
        }


if __name__ == '__main__':
    # 配置参数
    config = Configs().get_config()
    MODEL1_PATH = r'E:\GXU\cv\Face Detection\mywork\520.pth'
    MODEL2_PATH = 'weights/520_v3.pth'
    REF_IMAGE = 'me_247.jpg'
    TEST_IMAGE = 'test.png'

    # 创建识别器（设置不同阈值）
    recognizer = DualModelFaceRecognizer(
        config,
        MODEL1_PATH,
        MODEL2_PATH,
        REF_IMAGE,
        threshold1=0.80,  # 模型1阈值
        threshold2=0.70  # 模型2阈值
    )

    # 进行预测
    result = recognizer.predict(TEST_IMAGE)

    # 打印结果
    print("\n双模型人脸识别结果:")
    print(
        f"模型1相似度: {result['model1_similarity']:.4f} (阈值: {result['model1_threshold']:.2f}) → {'✅ 本人' if result['model1_result'] else '❌ 非本人'}")
    print(
        f"模型2相似度: {result['model2_similarity']:.4f} (阈值: {result['model2_threshold']:.2f}) → {'✅ 本人' if result['model2_result'] else '❌ 非本人'}")
    print(f"最终判定: {'✅ 本人' if result['final_decision'] else '❌ 非本人'}")
