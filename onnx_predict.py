"""
ONNX模型推理脚本
使用ONNX Runtime对图像进行语义分割推理

使用示例：
    # 单图推理
    python onnx_predict.py --mode predict --image img/street.jpg

    # 批量推理
    python onnx_predict.py --mode batch --input_dir img/ --output_dir img_out/

    # FPS测试
    python onnx_predict.py --mode fps --image img/street.jpg --test_interval 100
"""
import argparse
import os
import time
import colorsys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


class ONNXPredictor:
    """ONNX模型推理器"""

    def __init__(self, onnx_path, num_classes=21, input_shape=(512, 512),
                 mix_type=0, use_gpu=False):
        """
        初始化ONNX推理器

        参数:
            onnx_path: ONNX模型路径
            num_classes: 类别数量+1
            input_shape: 输入图像尺寸 (高, 宽)
            mix_type: 可视化方式 (0=混合, 1=仅分割, 2=仅目标)
            use_gpu: 是否使用GPU
        """
        self.onnx_path = onnx_path
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.mix_type = mix_type

        # 配置ONNX Runtime
        print("=" * 60)
        print("初始化ONNX Runtime")
        print("=" * 60)
        print(f"模型路径: {onnx_path}")
        print(f"类别数量: {num_classes}")
        print(f"输入尺寸: {input_shape}")
        print(f"可视化方式: {mix_type}")

        # 选择执行提供器
        providers = []
        if use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                print("使用设备: GPU (CUDA)")
            else:
                print("⚠ CUDA不可用，回退到CPU")
                providers.append('CPUExecutionProvider')
        else:
            providers.append('CPUExecutionProvider')
            print("使用设备: CPU")

        # 创建推理会话
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"实际提供器: {self.session.get_providers()}")

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 初始化颜色映射
        self._init_colors()
        print("=" * 60)

    def _init_colors(self):
        """初始化类别颜色映射"""
        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128), (128, 64, 12)
            ]
        else:
            # 为更多类别生成颜色
            hsv_tuples = [(x / self.num_classes, 1., 1.)
                          for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255),
                                             int(x[1] * 255),
                                             int(x[2] * 255)),
                                  self.colors))

    def _preprocess(self, image):
        """
        图像预处理

        参数:
            image: PIL Image对象

        返回:
            preprocessed: 预处理后的numpy数组 [1, 3, H, W]
            nw: resize后的宽度
            nh: resize后的高度
        """
        # 转换为RGB
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            image = image
        else:
            image = image.convert('RGB')

        # Resize（保持比例，填充灰色）
        iw, ih = image.size
        w, h = self.input_shape[1], self.input_shape[0]
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        # 归一化和转换
        image_data = np.array(new_image, dtype=np.float32)
        image_data /= 255.0  # 归一化到[0, 1]
        image_data = np.transpose(image_data, (2, 0, 1))  # HWC -> CHW
        image_data = np.expand_dims(image_data, 0)  # 添加batch维度

        return image_data, nw, nh

    def _postprocess(self, output, original_size, nw, nh):
        """
        后处理

        参数:
            output: 模型输出 [1, num_classes, H, W]
            original_size: 原始图像尺寸 (宽, 高)
            nw: resize后的宽度
            nh: resize后的高度

        返回:
            pr: 分割结果 [原始高, 原始宽]
        """
        # Softmax
        pr = output[0]  # [num_classes, H, W]
        pr = np.transpose(pr, (1, 2, 0))  # CHW -> HWC
        pr = self._softmax(pr, axis=-1)

        # 裁剪灰条
        pr = pr[
            int((self.input_shape[0] - nh) // 2):int((self.input_shape[0] - nh) // 2 + nh),
            int((self.input_shape[1] - nw) // 2):int((self.input_shape[1] - nw) // 2 + nw)
        ]

        # Resize到原始尺寸
        pr = cv2.resize(pr, original_size, interpolation=cv2.INTER_LINEAR)

        # Argmax获取类别
        pr = pr.argmax(axis=-1)

        return pr

    @staticmethod
    def _softmax(x, axis=-1):
        """Softmax函数"""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _visualize(self, original_image, segmentation):
        """
        可视化分割结果

        参数:
            original_image: 原始PIL Image
            segmentation: 分割结果 [H, W]

        返回:
            result: 可视化结果PIL Image
        """
        original_array = np.array(original_image)
        h, w = segmentation.shape

        # 创建彩色分割图
        seg_img = np.reshape(
            np.array(self.colors, np.uint8)[np.reshape(segmentation, [-1])],
            [h, w, -1]
        )

        if self.mix_type == 0:
            # 原图与分割图混合
            seg_pil = Image.fromarray(np.uint8(seg_img))
            result = Image.blend(original_image, seg_pil, 0.7)

        elif self.mix_type == 1:
            # 仅分割图
            result = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            # 仅保留目标（移除背景）
            seg_img = (np.expand_dims(segmentation != 0, -1) *
                      original_array.astype(np.float32)).astype('uint8')
            result = Image.fromarray(np.uint8(seg_img))

        else:
            result = original_image

        return result

    def predict(self, image_path, save_path=None, show=False):
        """
        预测单张图像

        参数:
            image_path: 图像路径
            save_path: 保存路径（可选）
            show: 是否显示结果

        返回:
            result_image: 可视化结果PIL Image
        """
        # 加载图像
        image = Image.open(image_path)
        original_size = image.size  # (宽, 高)

        # 预处理
        input_data, nw, nh = self._preprocess(image)

        # 推理
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )[0]

        # 后处理
        segmentation = self._postprocess(output, original_size, nw, nh)

        # 可视化
        result_image = self._visualize(image, segmentation)

        # 保存
        if save_path:
            result_image.save(save_path)
            print(f"结果已保存至: {save_path}")

        # 显示
        if show:
            result_image.show()

        return result_image

    def predict_batch(self, input_dir, output_dir, image_extensions=None):
        """
        批量预测

        参数:
            input_dir: 输入图像目录
            output_dir: 输出目录
            image_extensions: 图像文件扩展名（默认常用格式）
        """
        if image_extensions is None:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

        if len(image_files) == 0:
            print(f"⚠ 在 {input_dir} 中未找到图像文件")
            return

        print(f"\n找到 {len(image_files)} 张图像")
        print("=" * 60)

        # 批量处理
        for image_path in tqdm(image_files, desc="处理进度"):
            output_path = os.path.join(output_dir, image_path.name)
            try:
                self.predict(str(image_path), save_path=output_path, show=False)
            except Exception as e:
                print(f"\n✗ 处理失败 {image_path.name}: {str(e)}")

        print("=" * 60)
        print(f"✓ 批量处理完成，结果保存在: {output_dir}")

    def benchmark_fps(self, image_path, test_interval=100):
        """
        FPS性能测试

        参数:
            image_path: 测试图像路径
            test_interval: 测试次数

        返回:
            fps: 平均FPS
            latency: 平均延迟(秒)
        """
        print("\n" + "=" * 60)
        print("FPS性能测试")
        print("=" * 60)
        print(f"测试图像: {image_path}")
        print(f"测试次数: {test_interval}")

        # 加载和预处理
        image = Image.open(image_path)
        input_data, _, _ = self._preprocess(image)

        # 预热
        print("\n预热中...")
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: input_data})

        # 测试
        print("测试中...")
        start_time = time.time()
        for _ in tqdm(range(test_interval), desc="推理进度"):
            self.session.run([self.output_name], {self.input_name: input_data})
        end_time = time.time()

        # 计算结果
        total_time = end_time - start_time
        latency = total_time / test_interval
        fps = 1.0 / latency

        print("=" * 60)
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均延迟: {latency * 1000:.2f} ms")
        print(f"平均FPS: {fps:.2f}")
        print("=" * 60)

        return fps, latency


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ONNX模型推理')

    # 模式选择
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'batch', 'fps'],
                        help='运行模式 (predict=单图, batch=批量, fps=性能测试)')

    # 模型配置
    parser.add_argument('--onnx_path', type=str,
                        default='model_data/unet_vgg_voc.onnx',
                        help='ONNX模型路径')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='类别数量+1')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[512, 512],
                        help='输入尺寸 [高 宽]')

    # 推理配置
    parser.add_argument('--use_gpu', action='store_true',
                        help='使用GPU推理')
    parser.add_argument('--mix_type', type=int, default=0, choices=[0, 1, 2],
                        help='可视化方式 (0=混合, 1=仅分割, 2=仅目标)')

    # 单图预测
    parser.add_argument('--image', type=str, default='',
                        help='输入图像路径 (用于predict和fps模式)')
    parser.add_argument('--output', type=str, default='',
                        help='输出路径 (用于predict模式)')
    parser.add_argument('--show', action='store_true',
                        help='显示结果 (用于predict模式)')

    # 批量预测
    parser.add_argument('--input_dir', type=str, default='img/',
                        help='输入目录 (用于batch模式)')
    parser.add_argument('--output_dir', type=str, default='img_out/',
                        help='输出目录 (用于batch模式)')

    # FPS测试
    parser.add_argument('--test_interval', type=int, default=100,
                        help='测试次数 (用于fps模式)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查模型文件
    if not os.path.exists(args.onnx_path):
        print(f"✗ 模型文件不存在: {args.onnx_path}")
        print("\n请先使用 export_to_onnx.py 导出ONNX模型")
        return 1

    # 创建推理器
    try:
        predictor = ONNXPredictor(
            onnx_path=args.onnx_path,
            num_classes=args.num_classes,
            input_shape=tuple(args.input_shape),
            mix_type=args.mix_type,
            use_gpu=args.use_gpu
        )
    except Exception as e:
        print(f"✗ 初始化失败: {str(e)}")
        return 1

    # 执行对应模式
    try:
        if args.mode == 'predict':
            if not args.image:
                # 交互式输入
                while True:
                    image_path = input('\n请输入图像路径 (或输入 q 退出): ').strip()
                    if image_path.lower() == 'q':
                        break
                    if not os.path.exists(image_path):
                        print(f"✗ 文件不存在: {image_path}")
                        continue

                    predictor.predict(image_path, show=True)
            else:
                # 命令行模式
                if not os.path.exists(args.image):
                    print(f"✗ 文件不存在: {args.image}")
                    return 1

                predictor.predict(
                    args.image,
                    save_path=args.output if args.output else None,
                    show=args.show
                )

        elif args.mode == 'batch':
            if not os.path.exists(args.input_dir):
                print(f"✗ 输入目录不存在: {args.input_dir}")
                return 1

            predictor.predict_batch(args.input_dir, args.output_dir)

        elif args.mode == 'fps':
            if not args.image:
                print("✗ fps模式需要指定 --image 参数")
                return 1
            if not os.path.exists(args.image):
                print(f"✗ 文件不存在: {args.image}")
                return 1

            predictor.benchmark_fps(args.image, args.test_interval)

    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 0
    except Exception as e:
        print(f"\n✗ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
