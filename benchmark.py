"""
性能基准测试脚本
对比PyTorch和ONNX Runtime的推理性能

使用示例：
    python benchmark.py --pytorch_model model_data/unet_vgg_voc.pth --onnx_model model_data/unet_vgg_voc.onnx
"""
import argparse
import time
import os
import psutil
import platform
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

from nets.unet import Unet as UnetModel
from utils.utils import cvtColor, preprocess_input, resize_image


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self, pytorch_model_path, onnx_model_path, backbone, num_classes):
        """
        初始化基准测试器

        参数:
            pytorch_model_path: PyTorch模型路径
            onnx_model_path: ONNX模型路径
            backbone: 骨干网络类型
            num_classes: 类别数量
        """
        self.pytorch_model_path = pytorch_model_path
        self.onnx_model_path = onnx_model_path
        self.backbone = backbone
        self.num_classes = num_classes

        print("=" * 80)
        print("性能基准测试 - PyTorch vs ONNX Runtime")
        print("=" * 80)

        # 加载PyTorch模型
        print("\n[1/2] 加载PyTorch模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_model = UnetModel(num_classes=num_classes, backbone=backbone)
        self.pytorch_model.load_state_dict(
            torch.load(pytorch_model_path, map_location=self.device)
        )
        self.pytorch_model = self.pytorch_model.eval().to(self.device)
        print(f"  ✓ PyTorch模型已加载 (设备: {self.device})")

        # 加载ONNX模型
        print("\n[2/2] 加载ONNX模型...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name
        print(f"  ✓ ONNX模型已加载 (提供器: {self.onnx_session.get_providers()})")

        print("=" * 80)

        # 系统信息
        self._print_system_info()

    def _print_system_info(self):
        """打印系统信息"""
        print("\n系统信息:")
        print(f"  - 操作系统: {platform.system()} {platform.release()}")
        print(f"  - Python版本: {platform.python_version()}")
        print(f"  - PyTorch版本: {torch.__version__}")
        print(f"  - ONNX Runtime版本: {ort.__version__}")
        print(f"  - CPU: {platform.processor()}")
        print(f"  - CPU核心数: {psutil.cpu_count(logical=False)} 物理核心, {psutil.cpu_count(logical=True)} 逻辑核心")
        print(f"  - 内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")

        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("  - GPU: 不可用")

    def _preprocess(self, image, input_shape):
        """预处理图像（与原始代码一致）"""
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )
        return image_data

    def _pytorch_inference(self, image_data):
        """PyTorch推理"""
        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.device)
            output = self.pytorch_model(images)[0]
            pr = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr.argmax(axis=-1)
        return pr

    def _onnx_inference(self, image_data):
        """ONNX Runtime推理"""
        output = self.onnx_session.run(
            [self.onnx_output_name],
            {self.onnx_input_name: image_data}
        )[0][0]

        # Softmax
        pr = np.transpose(output, (1, 2, 0))
        pr = pr - np.max(pr, axis=-1, keepdims=True)
        exp_pr = np.exp(pr)
        pr = exp_pr / np.sum(exp_pr, axis=-1, keepdims=True)
        pr = pr.argmax(axis=-1)

        return pr

    def test_inference_speed(self, image_path, input_shapes, warmup=10, iterations=100):
        """
        测试推理速度

        参数:
            image_path: 测试图像路径
            input_shapes: 输入尺寸列表 [(H, W), ...]
            warmup: 预热次数
            iterations: 测试迭代次数

        返回:
            results: 测试结果字典
        """
        print("\n" + "=" * 80)
        print("推理速度测试")
        print("=" * 80)
        print(f"测试图像: {image_path}")
        print(f"预热次数: {warmup}")
        print(f"测试迭代: {iterations}")

        image = Image.open(image_path)
        results = defaultdict(dict)

        for input_shape in input_shapes:
            print(f"\n输入尺寸: {input_shape[0]}x{input_shape[1]}")
            print("-" * 80)

            # 预处理
            image_data = self._preprocess(image, input_shape)

            # PyTorch测试
            print("  [1/2] 测试PyTorch...")

            # 预热
            for _ in range(warmup):
                self._pytorch_inference(image_data)

            # 测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            pytorch_times = []
            for _ in tqdm(range(iterations), desc="    PyTorch推理", leave=False):
                start = time.perf_counter()
                self._pytorch_inference(image_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                pytorch_times.append(end - start)

            pytorch_mean = np.mean(pytorch_times)
            pytorch_std = np.std(pytorch_times)
            pytorch_fps = 1.0 / pytorch_mean

            print(f"    ✓ 平均延迟: {pytorch_mean*1000:.2f} ± {pytorch_std*1000:.2f} ms")
            print(f"    ✓ FPS: {pytorch_fps:.2f}")

            # ONNX测试
            print("  [2/2] 测试ONNX Runtime...")

            # 预热
            for _ in range(warmup):
                self._onnx_inference(image_data)

            # 测试
            onnx_times = []
            for _ in tqdm(range(iterations), desc="    ONNX推理", leave=False):
                start = time.perf_counter()
                self._onnx_inference(image_data)
                end = time.perf_counter()
                onnx_times.append(end - start)

            onnx_mean = np.mean(onnx_times)
            onnx_std = np.std(onnx_times)
            onnx_fps = 1.0 / onnx_mean

            print(f"    ✓ 平均延迟: {onnx_mean*1000:.2f} ± {onnx_std*1000:.2f} ms")
            print(f"    ✓ FPS: {onnx_fps:.2f}")

            # 计算加速比
            speedup = pytorch_mean / onnx_mean
            print(f"\n  加速比: {speedup:.2f}x ({'ONNX更快' if speedup > 1 else 'PyTorch更快'})")

            # 保存结果
            shape_key = f"{input_shape[0]}x{input_shape[1]}"
            results[shape_key] = {
                'pytorch': {
                    'mean_latency': pytorch_mean,
                    'std_latency': pytorch_std,
                    'fps': pytorch_fps
                },
                'onnx': {
                    'mean_latency': onnx_mean,
                    'std_latency': onnx_std,
                    'fps': onnx_fps
                },
                'speedup': speedup
            }

        return results

    def test_memory_usage(self, image_path, input_shape):
        """
        测试内存使用

        参数:
            image_path: 测试图像路径
            input_shape: 输入尺寸 (H, W)

        返回:
            memory_results: 内存使用结果
        """
        print("\n" + "=" * 80)
        print("内存使用测试")
        print("=" * 80)
        print(f"测试图像: {image_path}")
        print(f"输入尺寸: {input_shape[0]}x{input_shape[1]}")

        image = Image.open(image_path)
        image_data = self._preprocess(image, input_shape)

        results = {}

        # PyTorch内存测试
        print("\n[1/2] 测试PyTorch内存...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            self._pytorch_inference(image_data)
            torch.cuda.synchronize()

            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            print(f"  ✓ GPU内存使用: {gpu_memory:.2f} MB")
            results['pytorch_gpu'] = gpu_memory
        else:
            print("  - GPU不可用，跳过GPU内存测试")

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2)  # MB
        for _ in range(10):
            self._pytorch_inference(image_data)
        mem_after = process.memory_info().rss / (1024 ** 2)  # MB
        cpu_memory = mem_after - mem_before
        print(f"  ✓ CPU内存增量: {cpu_memory:.2f} MB")
        results['pytorch_cpu'] = cpu_memory

        # ONNX内存测试
        print("\n[2/2] 测试ONNX Runtime内存...")
        mem_before = process.memory_info().rss / (1024 ** 2)
        for _ in range(10):
            self._onnx_inference(image_data)
        mem_after = process.memory_info().rss / (1024 ** 2)
        cpu_memory = mem_after - mem_before
        print(f"  ✓ CPU内存增量: {cpu_memory:.2f} MB")
        results['onnx_cpu'] = cpu_memory

        return results

    def test_accuracy(self, image_path, input_shape):
        """
        测试输出一致性

        参数:
            image_path: 测试图像路径
            input_shape: 输入尺寸 (H, W)

        返回:
            accuracy_results: 准确性结果
        """
        print("\n" + "=" * 80)
        print("输出一致性测试")
        print("=" * 80)
        print(f"测试图像: {image_path}")
        print(f"输入尺寸: {input_shape[0]}x{input_shape[1]}")

        image = Image.open(image_path)
        image_data = self._preprocess(image, input_shape)

        # PyTorch推理
        print("\n[1/2] PyTorch推理...")
        pytorch_output = self._pytorch_inference(image_data)

        # ONNX推理
        print("[2/2] ONNX Runtime推理...")
        onnx_output = self._onnx_inference(image_data)

        # 计算差异
        print("\n[3/3] 计算差异...")
        pixel_diff = np.sum(pytorch_output != onnx_output)
        total_pixels = pytorch_output.size
        pixel_agreement = (total_pixels - pixel_diff) / total_pixels * 100

        print(f"  - 总像素数: {total_pixels}")
        print(f"  - 不一致像素: {pixel_diff}")
        print(f"  - 像素一致率: {pixel_agreement:.4f}%")

        if pixel_agreement > 99.9:
            print("  ✓ 输出一致性: 优秀")
        elif pixel_agreement > 99.0:
            print("  ✓ 输出一致性: 良好")
        elif pixel_agreement > 95.0:
            print("  ⚠ 输出一致性: 可接受")
        else:
            print("  ✗ 输出一致性: 较差")

        return {
            'pixel_agreement': pixel_agreement,
            'pixel_diff': pixel_diff,
            'total_pixels': total_pixels
        }

    def generate_report(self, speed_results, memory_results, accuracy_results, output_path='benchmark_report.txt'):
        """
        生成测试报告

        参数:
            speed_results: 速度测试结果
            memory_results: 内存测试结果
            accuracy_results: 准确性测试结果
            output_path: 报告保存路径
        """
        print("\n" + "=" * 80)
        print("生成测试报告")
        print("=" * 80)

        report = []
        report.append("=" * 80)
        report.append("U-Net 性能基准测试报告")
        report.append("=" * 80)
        report.append(f"\nPyTorch模型: {self.pytorch_model_path}")
        report.append(f"ONNX模型: {self.onnx_model_path}")
        report.append(f"骨干网络: {self.backbone}")
        report.append(f"类别数量: {self.num_classes}")

        # 速度测试结果
        report.append("\n" + "=" * 80)
        report.append("推理速度测试")
        report.append("=" * 80)

        for shape_key, results in speed_results.items():
            report.append(f"\n输入尺寸: {shape_key}")
            report.append("-" * 80)
            report.append(f"{'框架':<20} {'延迟(ms)':<15} {'FPS':<15}")
            report.append("-" * 80)

            pytorch_latency = results['pytorch']['mean_latency'] * 1000
            pytorch_fps = results['pytorch']['fps']
            report.append(f"{'PyTorch':<20} {pytorch_latency:<15.2f} {pytorch_fps:<15.2f}")

            onnx_latency = results['onnx']['mean_latency'] * 1000
            onnx_fps = results['onnx']['fps']
            report.append(f"{'ONNX Runtime':<20} {onnx_latency:<15.2f} {onnx_fps:<15.2f}")

            report.append("-" * 80)
            report.append(f"加速比: {results['speedup']:.2f}x")

        # 内存使用结果
        report.append("\n" + "=" * 80)
        report.append("内存使用")
        report.append("=" * 80)
        report.append(f"\n{'框架':<20} {'GPU内存(MB)':<20} {'CPU内存增量(MB)':<20}")
        report.append("-" * 80)

        pytorch_gpu = memory_results.get('pytorch_gpu', 'N/A')
        pytorch_cpu = memory_results.get('pytorch_cpu', 0)
        report.append(f"{'PyTorch':<20} {pytorch_gpu if isinstance(pytorch_gpu, str) else f'{pytorch_gpu:.2f}':<20} {pytorch_cpu:<20.2f}")

        onnx_gpu = memory_results.get('onnx_gpu', 'N/A')
        onnx_cpu = memory_results.get('onnx_cpu', 0)
        report.append(f"{'ONNX Runtime':<20} {onnx_gpu if isinstance(onnx_gpu, str) else f'{onnx_gpu:.2f}':<20} {onnx_cpu:<20.2f}")

        # 准确性结果
        report.append("\n" + "=" * 80)
        report.append("输出一致性")
        report.append("=" * 80)
        report.append(f"\n像素一致率: {accuracy_results['pixel_agreement']:.4f}%")
        report.append(f"不一致像素: {accuracy_results['pixel_diff']} / {accuracy_results['total_pixels']}")

        report.append("\n" + "=" * 80)

        # 打印并保存报告
        report_text = "\n".join(report)
        print(report_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✓ 报告已保存至: {output_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='性能基准测试 - PyTorch vs ONNX Runtime')

    parser.add_argument('--pytorch_model', type=str, required=True,
                        help='PyTorch模型路径')
    parser.add_argument('--onnx_model', type=str, required=True,
                        help='ONNX模型路径')
    parser.add_argument('--backbone', type=str, default='vgg',
                        choices=['vgg', 'resnet50'],
                        help='骨干网络类型')
    parser.add_argument('--num_classes', type=int, default=21,
                        help='类别数量+1')

    parser.add_argument('--image', type=str, default='img/street.jpg',
                        help='测试图像路径')
    parser.add_argument('--input_shapes', type=int, nargs='+',
                        default=[256, 512, 1024],
                        help='测试的输入尺寸列表 (例如: 256 512 1024)')

    parser.add_argument('--warmup', type=int, default=10,
                        help='预热次数')
    parser.add_argument('--iterations', type=int, default=100,
                        help='测试迭代次数')

    parser.add_argument('--output', type=str, default='benchmark_report.txt',
                        help='报告保存路径')

    return parser.parse_args()


def main():
    args = parse_args()

    # 检查文件
    if not os.path.exists(args.pytorch_model):
        print(f"✗ PyTorch模型不存在: {args.pytorch_model}")
        return 1

    if not os.path.exists(args.onnx_model):
        print(f"✗ ONNX模型不存在: {args.onnx_model}")
        return 1

    if not os.path.exists(args.image):
        print(f"✗ 测试图像不存在: {args.image}")
        return 1

    # 创建基准测试器
    benchmark = PerformanceBenchmark(
        pytorch_model_path=args.pytorch_model,
        onnx_model_path=args.onnx_model,
        backbone=args.backbone,
        num_classes=args.num_classes
    )

    # 转换输入尺寸
    input_shapes = [(size, size) for size in args.input_shapes]

    try:
        # 速度测试
        speed_results = benchmark.test_inference_speed(
            image_path=args.image,
            input_shapes=input_shapes,
            warmup=args.warmup,
            iterations=args.iterations
        )

        # 内存测试（使用第一个尺寸）
        memory_results = benchmark.test_memory_usage(
            image_path=args.image,
            input_shape=input_shapes[0]
        )

        # 准确性测试（使用第一个尺寸）
        accuracy_results = benchmark.test_accuracy(
            image_path=args.image,
            input_shape=input_shapes[0]
        )

        # 生成报告
        benchmark.generate_report(
            speed_results=speed_results,
            memory_results=memory_results,
            accuracy_results=accuracy_results,
            output_path=args.output
        )

        print("\n✓ 基准测试完成！")

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
