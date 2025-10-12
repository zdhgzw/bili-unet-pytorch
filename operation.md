# U-Net 模型 ONNX 部署完整操作手册

本文档提供 U-Net PyTorch 模型导出为 ONNX 格式并进行推理部署的完整步骤。

## 目录
- [前提条件](#前提条件)
- [步骤 1：安装依赖](#步骤-1安装依赖)
- [步骤 2：准备权重文件](#步骤-2准备权重文件)
- [步骤 3：导出 ONNX 模型](#步骤-3导出-onnx-模型)
- [步骤 4：ONNX 模型推理](#步骤-4onnx-模型推理)
- [参数说明](#参数说明)
- [可视化模式](#可视化模式)
- [完整示例](#完整示例)
- [常见问题](#常见问题)

---

## 前提条件

**模型信息：**
- 权重文件：`unet_vgg_voc.pth`
- 骨干网络：VGG
- 类别数量：21（VOC数据集：20类+背景）
- 输入尺寸：512×512

**Python 环境：**
- Python 3.7+
- PyTorch 1.7+

---

## 步骤 1：安装依赖

安装 ONNX 相关依赖包：

```bash
# 基础依赖（必需）
pip install onnx onnxruntime

# 模型简化工具（可选，推荐）
pip install onnx-simplifier

# GPU 推理支持（可选，需要 CUDA）
pip install onnxruntime-gpu
```

**验证安装：**
```bash
python -c "import onnx; import onnxruntime; print('ONNX安装成功')"
```

---

## 步骤 2：准备权重文件

将下载的权重文件 `unet_vgg_voc.pth` 放到项目的 `model_data/` 目录下：

```bash
# 确保目录结构如下
unet-pytorch/
├── model_data/
│   └── unet_vgg_voc.pth  # 将权重文件放在这里
├── export_to_onnx.py
├── onnx_predict.py
└── ...
```

**验证文件：**
```bash
# Windows
dir model_data\unet_vgg_voc.pth

# Linux/Mac
ls -lh model_data/unet_vgg_voc.pth
```

---

## 步骤 3：导出 ONNX 模型

使用 `export_to_onnx.py` 脚本将 PyTorch 模型导出为 ONNX 格式。

### 3.1 基础导出（推荐）

```bash
python export_to_onnx.py \
    --model_path model_data/unet_vgg_voc.pth \
    --backbone vgg \
    --num_classes 21 \
    --input_shape 512 512 \
    --onnx_path model_data/unet_vgg_voc.onnx
```

### 3.2 带模型简化（推荐，需要 onnx-simplifier）

```bash
python export_to_onnx.py \
    --model_path model_data/unet_vgg_voc.pth \
    --backbone vgg \
    --num_classes 21 \
    --input_shape 512 512 \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --simplify
```

### 3.3 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_path` | PyTorch 权重文件路径 | `model_data/unet_vgg_voc.pth` |
| `--backbone` | 骨干网络类型 | `vgg` 或 `resnet50` |
| `--num_classes` | 类别数量+1 | `21`（VOC）、`2`（二分类） |
| `--input_shape` | 输入图像尺寸 [高 宽] | `512 512` |
| `--onnx_path` | ONNX 模型保存路径 | `model_data/unet_vgg_voc.onnx` |
| `--opset_version` | ONNX opset 版本（可选） | `12`（默认） |
| `--simplify` | 是否简化模型（可选） | 添加此参数启用 |
| `--verify` | 是否验证模型（可选） | 默认开启 |

### 3.4 预期输出

```
===========================================================
ONNX模型导出工具
===========================================================

[1/5] 加载PyTorch模型...
  - 模型路径: model_data/unet_vgg_voc.pth
  - 骨干网络: vgg
  - 类别数量: 21
  - 输入尺寸: [512, 512]
  ✓ 模型加载成功

[2/5] 准备输入张量...
  - 输入shape: torch.Size([1, 3, 512, 512]) (NCHW)

[3/5] 导出ONNX模型...
  - 保存路径: model_data/unet_vgg_voc.onnx
  - Opset版本: 12
  ✓ ONNX导出成功

[4/5] 检查ONNX模型...
  ✓ ONNX模型检查通过

[5/5] 简化ONNX模型...
  ✓ 模型简化成功

===========================================================
验证ONNX模型
===========================================================

[1/3] 加载ONNX Runtime...
  - 提供器: ['CPUExecutionProvider']

[2/3] 加载PyTorch模型进行对比...

[3/3] 对比输出...
  - 最大差异: 0.00000xxx
  - 平均差异: 0.00000xxx
  ✓ 验证通过：输出一致性良好

===========================================================
模型信息
===========================================================

文件大小: XX.XX MB

输入信息:
  - 名称: images
  - 形状: [1, 3, 512, 512]
  - 类型: 1

输出信息:
  - 名称: output
  - 形状: [1, 21, 512, 512]
  - 类型: 1

===========================================================
ONNX模型已保存至: model_data/unet_vgg_voc.onnx
===========================================================

✓ 导出完成！
```

---

## 步骤 4：ONNX 模型推理

使用 `onnx_predict.py` 脚本进行推理。支持三种模式：
- **predict**：单图推理
- **batch**：批量推理
- **fps**：性能测试

### 4.1 单图推理（交互式）

```bash
python onnx_predict.py --mode predict
```

然后根据提示输入图像路径：
```
请输入图像路径 (或输入 q 退出): img/street.jpg
```

### 4.2 单图推理（命令行）

```bash
python onnx_predict.py \
    --mode predict \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --num_classes 21 \
    --image img/street.jpg \
    --output img_out/street_result.jpg \
    --show
```

### 4.3 批量推理

处理文件夹中的所有图像：

```bash
python onnx_predict.py \
    --mode batch \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --num_classes 21 \
    --input_dir img/ \
    --output_dir img_out/
```

### 4.4 FPS 性能测试

#### CPU 推理性能测试

```bash
python onnx_predict.py \
    --mode fps \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --num_classes 21 \
    --image img/street.jpg \
    --test_interval 100
```

#### GPU 推理性能测试

需要安装 `onnxruntime-gpu` 并且有 CUDA 支持：

```bash
python onnx_predict.py \
    --mode fps \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --num_classes 21 \
    --image img/street.jpg \
    --test_interval 100 \
    --use_gpu
```

**预期输出：**
```
===========================================================
FPS性能测试
===========================================================
测试图像: img/street.jpg
测试次数: 100

预热中...
测试中...
推理进度: 100%|████████████████████| 100/100

===========================================================
总耗时: X.XX 秒
平均延迟: XX.XX ms
平均FPS: XX.XX
===========================================================
```

---

## 参数说明

### 推理参数完整列表

| 参数 | 说明 | 默认值 | 适用模式 |
|------|------|--------|----------|
| `--mode` | 运行模式 | `predict` | 所有 |
| `--onnx_path` | ONNX模型路径 | `model_data/unet_vgg_voc.onnx` | 所有 |
| `--num_classes` | 类别数量+1 | `21` | 所有 |
| `--input_shape` | 输入尺寸 [高 宽] | `[512, 512]` | 所有 |
| `--use_gpu` | 使用GPU推理 | `False` | 所有 |
| `--mix_type` | 可视化方式（0/1/2） | `0` | predict, batch |
| `--image` | 输入图像路径 | - | predict, fps |
| `--output` | 输出路径 | - | predict |
| `--show` | 显示结果 | `False` | predict |
| `--input_dir` | 输入目录 | `img/` | batch |
| `--output_dir` | 输出目录 | `img_out/` | batch |
| `--test_interval` | 测试次数 | `100` | fps |

---

## 可视化模式

通过 `--mix_type` 参数控制分割结果的可视化方式：

### mix_type=0（混合模式，默认）
原图与分割结果按 0.7 比例混合，可同时看到原图和分割效果。

```bash
python onnx_predict.py --mode predict --image img/street.jpg --mix_type 0
```

### mix_type=1（仅分割）
仅显示彩色分割掩码，适合查看纯分割结果。

```bash
python onnx_predict.py --mode predict --image img/street.jpg --mix_type 1
```

### mix_type=2（仅目标）
移除背景，仅保留分割目标，适合抠图应用。

```bash
python onnx_predict.py --mode predict --image img/street.jpg --mix_type 2
```

---

## 完整示例

### 示例 1：从头到尾的完整流程

```bash
# 1. 安装依赖
pip install onnx onnxruntime onnx-simplifier

# 2. 确认权重文件位置
# 将 unet_vgg_voc.pth 放到 model_data/ 目录

# 3. 导出 ONNX 模型（带简化）
python export_to_onnx.py \
    --model_path model_data/unet_vgg_voc.pth \
    --backbone vgg \
    --num_classes 21 \
    --input_shape 512 512 \
    --onnx_path model_data/unet_vgg_voc.onnx \
    --simplify

# 4. 单图推理测试
python onnx_predict.py \
    --mode predict \
    --image img/street.jpg \
    --output img_out/result.jpg \
    --show

# 5. 批量推理
python onnx_predict.py \
    --mode batch \
    --input_dir img/ \
    --output_dir img_out/

# 6. 性能测试
python onnx_predict.py \
    --mode fps \
    --image img/street.jpg \
    --test_interval 100
```

### 示例 2：自定义数据集部署

假设你有一个二分类分割模型（背景+目标）：

```bash
# 1. 导出 ONNX（num_classes=2）
python export_to_onnx.py \
    --model_path logs/best_epoch_weights.pth \
    --backbone vgg \
    --num_classes 2 \
    --input_shape 512 512 \
    --onnx_path model_data/custom_model.onnx \
    --simplify

# 2. 推理
python onnx_predict.py \
    --mode predict \
    --onnx_path model_data/custom_model.onnx \
    --num_classes 2 \
    --image test.jpg \
    --output result.jpg \
    --mix_type 2  # 仅保留目标
```

### 示例 3：ResNet50 骨干网络

```bash
# 1. 导出
python export_to_onnx.py \
    --model_path model_data/unet_resnet50_voc.pth \
    --backbone resnet50 \
    --num_classes 21 \
    --input_shape 512 512 \
    --onnx_path model_data/unet_resnet50_voc.onnx \
    --simplify

# 2. 推理
python onnx_predict.py \
    --onnx_path model_data/unet_resnet50_voc.onnx \
    --num_classes 21 \
    --image img/street.jpg \
    --show
```

---

## 常见问题

### Q1: 导出时报错 "模型文件不存在"
**解决方法：**
- 确认权重文件路径正确
- 确认文件名拼写正确（区分大小写）
- 使用绝对路径尝试

### Q2: 推理时输出全黑或全白
**可能原因：**
- `num_classes` 参数与训练时不一致
- `backbone` 参数与训练时不一致
- 模型权重与配置不匹配

**解决方法：**
确保导出和推理时的参数完全一致：
```bash
# 导出时
--backbone vgg --num_classes 21

# 推理时也必须
--num_classes 21
```

### Q3: GPU 推理不生效
**解决方法：**
```bash
# 1. 检查是否安装 GPU 版本
pip install onnxruntime-gpu

# 2. 检查 CUDA 是否可用
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 应该看到 'CUDAExecutionProvider'

# 3. 使用 --use_gpu 参数
python onnx_predict.py --mode predict --image test.jpg --use_gpu
```

### Q4: 验证时输出差异较大
**解决方法：**
- 小差异（< 1e-3）是正常的，可以接受
- 大差异可能是 opset 版本问题，尝试更换版本：
```bash
python export_to_onnx.py ... --opset_version 11
```

### Q5: 如何更改输入尺寸？
**注意：** 必须在导出时指定，推理时必须使用相同尺寸。

```bash
# 导出为 640×640
python export_to_onnx.py \
    --model_path model_data/unet_vgg_voc.pth \
    --backbone vgg \
    --num_classes 21 \
    --input_shape 640 640 \
    --onnx_path model_data/unet_vgg_voc_640.onnx

# 推理时也用 640×640
python onnx_predict.py \
    --onnx_path model_data/unet_vgg_voc_640.onnx \
    --input_shape 640 640 \
    --image test.jpg
```

### Q6: 批量推理找不到图像
**解决方法：**
- 确认输入目录路径正确
- 支持的格式：`.jpg, .jpeg, .png, .bmp, .tif, .tiff`
- 检查文件扩展名大小写

---

## 性能优化建议

### 1. 模型简化
使用 `--simplify` 参数可以减小模型大小并提升推理速度：
```bash
python export_to_onnx.py ... --simplify
```

### 2. GPU 加速
安装 GPU 版本并使用 `--use_gpu`：
```bash
pip install onnxruntime-gpu
python onnx_predict.py --mode fps --image test.jpg --use_gpu
```

### 3. 批量处理
批量推理比单张处理更高效：
```bash
python onnx_predict.py --mode batch --input_dir img/ --output_dir img_out/
```

### 4. TensorRT 加速（高级）
如果有 NVIDIA GPU，可以使用 TensorRT 进一步加速：
```bash
pip install onnxruntime-gpu tensorrt
# 需要额外配置 TensorRT 环境
```

---

## 参考资源

- **ONNX 官方文档**: https://onnx.ai/
- **ONNX Runtime 文档**: https://onnxruntime.ai/
- **PyTorch ONNX 导出指南**: https://pytorch.org/docs/stable/onnx.html
- **项目完整文档**: 参见 `CLAUDE.md`

---

## 总结

完整的 ONNX 部署流程：

1. ✅ 安装依赖：`pip install onnx onnxruntime onnx-simplifier`
2. ✅ 准备权重：将 `.pth` 文件放到 `model_data/`
3. ✅ 导出模型：`python export_to_onnx.py --model_path ... --backbone ... --num_classes ...`
4. ✅ 推理测试：`python onnx_predict.py --mode predict --image ...`
5. ✅ 批量部署：`python onnx_predict.py --mode batch --input_dir ... --output_dir ...`

关键要点：
- 导出和推理时的参数必须一致（`num_classes`、`backbone`、`input_shape`）
- 模型简化可以提升性能
- GPU 推理需要安装 `onnxruntime-gpu`
- 支持三种可视化模式（混合、仅分割、仅目标）

---

*最后更新时间：2025-10-12*
