"""
ONNX模型导出脚本
将训练好的U-Net PyTorch模型导出为ONNX格式

使用示例：
    python export_to_onnx.py --model_path logs/best_epoch_weights.pth --backbone vgg --num_classes 21
"""
import argparse
import os
import torch
import onnx
import onnxruntime
import numpy as np

from nets.unet import Unet


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='导出U-Net模型为ONNX格式')

    # 模型配置参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='PyTorch模型权重路径 (例如: logs/best_epoch_weights.pth)')
    parser.add_argument('--backbone', type=str, default='vgg', choices=['vgg', 'resnet50'],
                        help='骨干网络类型 (默认: vgg)')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='类别数量+1 (例如: 21表示20类+背景)')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[512, 512],
                        help='输入图像尺寸 [高 宽] (默认: 512 512)')

    # ONNX导出参数
    parser.add_argument('--onnx_path', type=str, default='',
                        help='ONNX模型保存路径 (默认: 与model_path同名.onnx)')
    parser.add_argument('--opset_version', type=int, default=12,
                        help='ONNX opset版本 (默认: 12)')
    parser.add_argument('--simplify', action='store_true',
                        help='是否使用onnx-simplifier简化模型')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='是否验证导出的ONNX模型 (默认: True)')

    return parser.parse_args()


def export_onnx(model_path, backbone, num_classes, input_shape,
                onnx_path, opset_version=12, simplify=False):
    """
    导出ONNX模型

    参数:
        model_path: PyTorch模型权重路径
        backbone: 骨干网络类型 ('vgg' 或 'resnet50')
        num_classes: 类别数量+1
        input_shape: 输入图像尺寸 [高, 宽]
        onnx_path: ONNX模型保存路径
        opset_version: ONNX opset版本
        simplify: 是否简化模型
    """
    print("=" * 60)
    print("ONNX模型导出工具")
    print("=" * 60)

    # 1. 加载PyTorch模型
    print(f"\n[1/5] 加载PyTorch模型...")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 骨干网络: {backbone}")
    print(f"  - 类别数量: {num_classes}")
    print(f"  - 输入尺寸: {input_shape}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型并加载权重
    device = torch.device('cpu')
    model = Unet(num_classes=num_classes, backbone=backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("  ✓ 模型加载成功")

    # 2. 准备dummy输入
    print(f"\n[2/5] 准备输入张量...")
    dummy_input = torch.zeros(1, 3, input_shape[0], input_shape[1]).to(device)
    print(f"  - 输入shape: {dummy_input.shape} (NCHW)")

    # 3. 导出ONNX模型
    print(f"\n[3/5] 导出ONNX模型...")
    print(f"  - 保存路径: {onnx_path}")
    print(f"  - Opset版本: {opset_version}")

    input_names = ["images"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=opset_version,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None
    )
    print("  ✓ ONNX导出成功")

    # 4. 检查ONNX模型
    print(f"\n[4/5] 检查ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX模型检查通过")

    # 5. 简化模型（可选）
    if simplify:
        print(f"\n[5/5] 简化ONNX模型...")
        try:
            import onnxsim
            print(f"  - onnx-simplifier版本: {onnxsim.__version__}")

            onnx_model, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=False,
                input_shapes=None
            )

            if not check:
                print("  ⚠ 模型简化失败，保存原始模型")
            else:
                onnx.save(onnx_model, onnx_path)
                print("  ✓ 模型简化成功")
        except ImportError:
            print("  ⚠ 未安装onnx-simplifier，跳过简化步骤")
            print("  提示: pip install onnx-simplifier")
    else:
        print(f"\n[5/5] 跳过模型简化")

    return onnx_path


def verify_onnx(onnx_path, model_path, backbone, num_classes, input_shape):
    """
    验证ONNX模型输出与PyTorch模型一致性

    参数:
        onnx_path: ONNX模型路径
        model_path: PyTorch模型路径
        backbone: 骨干网络类型
        num_classes: 类别数量
        input_shape: 输入尺寸
    """
    print("\n" + "=" * 60)
    print("验证ONNX模型")
    print("=" * 60)

    # 1. 加载ONNX模型
    print("\n[1/3] 加载ONNX Runtime...")
    ort_session = onnxruntime.InferenceSession(onnx_path)
    print(f"  - 提供器: {ort_session.get_providers()}")

    # 2. 加载PyTorch模型
    print("\n[2/3] 加载PyTorch模型进行对比...")
    device = torch.device('cpu')
    pytorch_model = Unet(num_classes=num_classes, backbone=backbone)
    pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
    pytorch_model.eval()

    # 3. 对比输出
    print("\n[3/3] 对比输出...")
    test_input = np.random.randn(1, 3, input_shape[0], input_shape[1]).astype(np.float32)

    # PyTorch推理
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input))[0].numpy()

    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    onnx_output = ort_session.run(None, ort_inputs)[0][0]

    # 计算差异
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  - 最大差异: {max_diff:.8f}")
    print(f"  - 平均差异: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print("  ✓ 验证通过：输出一致性良好")
    elif max_diff < 1e-3:
        print("  ⚠ 验证通过：输出存在轻微差异（可接受）")
    else:
        print("  ✗ 验证失败：输出差异较大")


def print_model_info(onnx_path):
    """打印模型信息"""
    print("\n" + "=" * 60)
    print("模型信息")
    print("=" * 60)

    # 文件大小
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
    print(f"\n文件大小: {file_size:.2f} MB")

    # 模型输入输出
    onnx_model = onnx.load(onnx_path)
    print("\n输入信息:")
    for input_tensor in onnx_model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  - 名称: {input_tensor.name}")
        print(f"  - 形状: {shape}")
        print(f"  - 类型: {input_tensor.type.tensor_type.elem_type}")

    print("\n输出信息:")
    for output_tensor in onnx_model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  - 名称: {output_tensor.name}")
        print(f"  - 形状: {shape}")
        print(f"  - 类型: {output_tensor.type.tensor_type.elem_type}")

    print("\n" + "=" * 60)
    print(f"ONNX模型已保存至: {onnx_path}")
    print("=" * 60)


def main():
    args = parse_args()

    # 如果未指定onnx_path，使用model_path同名文件
    if not args.onnx_path:
        base_name = os.path.splitext(args.model_path)[0]
        args.onnx_path = f"{base_name}.onnx"

    # 确保输出目录存在
    output_dir = os.path.dirname(args.onnx_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 导出ONNX模型
        onnx_path = export_onnx(
            model_path=args.model_path,
            backbone=args.backbone,
            num_classes=args.num_classes,
            input_shape=args.input_shape,
            onnx_path=args.onnx_path,
            opset_version=args.opset_version,
            simplify=args.simplify
        )

        # 验证模型
        if args.verify:
            verify_onnx(
                onnx_path=onnx_path,
                model_path=args.model_path,
                backbone=args.backbone,
                num_classes=args.num_classes,
                input_shape=args.input_shape
            )

        # 打印模型信息
        print_model_info(onnx_path)

        print("\n✓ 导出完成！")

    except Exception as e:
        print(f"\n✗ 导出失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
