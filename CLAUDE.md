# CLAUDE.md

## 语言规范
- 所有对话以简体中文回复
- 文档以简体中文格式输出，UTF-8格式

## 工作流程
1. 了解现状
2. 设计方案
3. 给出计划
4. 执行计划，每次只执行一个任务

每个流程执行后请等待review，收到OK后再继续下一任务


## 代码库项目概述

这是一个基于 PyTorch 的 U-Net 语义分割实现，主要用于生物医学图像分割，同时也支持 VOC 格式数据集。该实现包含多种骨干网络选项（VGG、ResNet50），支持标准数据集和医学影像数据集。

**重要提示**：U-Net 不太适合 VOC 类型数据集。它在需要浅层特征的少特征数据集上表现最佳，例如医学影像数据集。

## 关键命令

### 训练

**对于 VOC 数据集：**
```bash
python train.py
```

**对于医学数据集：**
```bash
python train_medical.py
```

**多GPU训练（仅限 Ubuntu）：**
- DP 模式：`CUDA_VISIBLE_DEVICES=0,1 python train.py`
- DDP 模式：在训练文件中设置 `distributed = True`，然后运行： 
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
  ```

### 预测

```bash
python predict.py
```
然后在提示时输入图像路径（例如：`img/street.jpg`）

### 评估（mIoU 计算）

```bash
python get_miou.py
```

### 数据集准备

**为 VOC 格式数据生成 train/val 划分：**
```bash
python voc_annotation.py
```

这将在 `VOCdevkit/VOC2007/ImageSets/Segmentation/` 中创建 train.txt、val.txt、trainval.txt 和 test.txt

### 模型架构检查

```bash
python summary.py
```

## 架构概述

### 核心组件

**网络结构（`nets/unet.py`）**
- `Unet`：主模型类，采用编码器-解码器架构
- 支持两种骨干网络：VGG16（`nets/vgg.py`）和 ResNet50（`nets/resnet.py`）
- 在编码器和解码器路径之间使用跳跃连接
- `unetUp`：上采样块，用于拼接来自编码器的特征

**预测类（`unet.py`）**
- `Unet`：用于推理的包装类，带有可配置参数
- `Unet_ONNX`：支持 ONNX 运行时推理
- 通过 `_defaults` 字典进行关键配置：model_path、num_classes、backbone、input_shape、cuda

**训练流程**
- 两阶段训练：冻结骨干网络（freeze）然后解冻（unfreeze）
- 冻结阶段仅训练解码器头部，节省内存并防止权重损坏
- 解冻阶段训练整个网络以实现完全适应
- 使用基于批次大小的自适应学习率

### 网络结构图

#### VGG16 骨干的 U-Net 架构

```
输入图像: 512×512×3
       ↓
┌──────────────── VGG16 编码器（下采样路径）────────────────┐
│                                                           │
│  [feat1] 512×512×64  ←─────────────────────┐            │
│      ↓ MaxPool                              │ 跳跃连接    │
│  [feat2] 256×256×128 ←─────────────────┐   │            │
│      ↓ MaxPool                          │   │            │
│  [feat3] 128×128×256 ←─────────────┐   │   │            │
│      ↓ MaxPool                      │   │   │            │
│  [feat4] 64×64×512   ←─────────┐   │   │   │            │
│      ↓ MaxPool                  │   │   │   │            │
│  [feat5] 32×32×512 (最底层)     │   │   │   │            │
│                                 │   │   │   │            │
└─────────────────────────────────┼───┼───┼───┼────────────┘
                                  ↓   │   │   │
┌──────────────── U-Net 解码器（上采样路径）───────────────┐
│                                 │   │   │   │            │
│  [up4] 拼接 → 64×64×1024       │   │   │   │            │
│        ↓ unetUp                 │   │   │   │            │
│        64×64×512 ───────────────┘   │   │   │            │
│        ↓ Upsample                   │   │   │            │
│  [up3] 拼接 → 128×128×768           │   │   │            │
│        ↓ unetUp                     │   │   │            │
│        128×128×256 ─────────────────┘   │   │            │
│        ↓ Upsample                       │   │            │
│  [up2] 拼接 → 256×256×384               │   │            │
│        ↓ unetUp                         │   │            │
│        256×256×128 ─────────────────────┘   │            │
│        ↓ Upsample                           │            │
│  [up1] 拼接 → 512×512×192                   │            │
│        ↓ unetUp                             │            │
│        512×512×64 ──────────────────────────┘            │
│        ↓ Conv 1×1                                        │
│  输出: 512×512×num_classes                               │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**各层详细参数（VGG16）：**

| 层级 | 编码器输出尺寸 | 解码器输入尺寸 | 解码器输出尺寸 | 说明 |
|------|---------------|---------------|---------------|------|
| feat1 | 512×512×64 | - | - | VGG features[:4] |
| feat2 | 256×256×128 | - | - | VGG features[4:9] |
| feat3 | 128×128×256 | - | - | VGG features[9:16] |
| feat4 | 64×64×512 | - | - | VGG features[16:23] |
| feat5 | 32×32×512 | - | - | VGG features[23:-1] |
| up4 | - | 64×64×1024 | 64×64×512 | 拼接(feat4 + upsample(feat5)) |
| up3 | - | 128×128×768 | 128×128×256 | 拼接(feat3 + upsample(up4)) |
| up2 | - | 256×256×384 | 256×256×128 | 拼接(feat2 + upsample(up3)) |
| up1 | - | 512×512×192 | 512×512×64 | 拼接(feat1 + upsample(up2)) |

#### ResNet50 骨干的 U-Net 架构

```
输入图像: 512×512×3
       ↓
┌──────────────── ResNet50 编码器（下采样路径）─────────────┐
│                                                           │
│  Conv7×7 (stride=2) + BN + ReLU                          │
│  [feat1] 256×256×64  ←─────────────────────┐            │
│      ↓ MaxPool (stride=2)                   │ 跳跃连接    │
│  [feat2] 128×128×256 (layer1) ←────────┐   │            │
│      ↓ layer2 (stride=2)                │   │            │
│  [feat3] 64×64×512   ←─────────────┐   │   │            │
│      ↓ layer3 (stride=2)            │   │   │            │
│  [feat4] 32×32×1024  ←─────────┐   │   │   │            │
│      ↓ layer4 (stride=2)        │   │   │   │            │
│  [feat5] 16×16×2048 (最底层)    │   │   │   │            │
│                                 │   │   │   │            │
└─────────────────────────────────┼───┼───┼───┼────────────┘
                                  ↓   │   │   │
┌──────────────── U-Net 解码器（上采样路径）───────────────┐
│                                 │   │   │   │            │
│  [up4] 拼接 → 32×32×3072       │   │   │   │            │
│        ↓ unetUp                 │   │   │   │            │
│        32×32×512 ───────────────┘   │   │   │            │
│        ↓ Upsample                   │   │   │            │
│  [up3] 拼接 → 64×64×1024            │   │   │            │
│        ↓ unetUp                     │   │   │            │
│        64×64×256 ────────────────────┘   │   │            │
│        ↓ Upsample                       │   │            │
│  [up2] 拼接 → 128×128×512               │   │            │
│        ↓ unetUp                         │   │            │
│        128×128×128 ──────────────────────┘   │            │
│        ↓ Upsample                           │            │
│  [up1] 拼接 → 256×256×192                   │            │
│        ↓ unetUp                             │            │
│        256×256×64 ──────────────────────────┘            │
│        ↓ Upsample×2 + Conv3×3×2                          │
│        512×512×64                                        │
│        ↓ Conv 1×1                                        │
│  输出: 512×512×num_classes                               │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**各层详细参数（ResNet50）：**

| 层级 | 编码器输出尺寸 | 解码器输入尺寸 | 解码器输出尺寸 | 说明 |
|------|---------------|---------------|---------------|------|
| feat1 | 256×256×64 | - | - | Conv7×7 + BN + ReLU |
| feat2 | 128×128×256 | - | - | MaxPool + layer1 (3个Bottleneck) |
| feat3 | 64×64×512 | - | - | layer2 (4个Bottleneck) |
| feat4 | 32×32×1024 | - | - | layer3 (6个Bottleneck) |
| feat5 | 16×16×2048 | - | - | layer4 (3个Bottleneck) |
| up4 | - | 32×32×3072 | 32×32×512 | 拼接(feat4 + upsample(feat5)) |
| up3 | - | 64×64×1024 | 64×64×256 | 拼接(feat3 + upsample(up4)) |
| up2 | - | 128×128×512 | 128×128×128 | 拼接(feat2 + upsample(up3)) |
| up1 | - | 256×256×192 | 256×256×64 | 拼接(feat1 + upsample(up2)) |
| up_conv | - | 256×256×64 | 512×512×64 | 双线性上采样×2 + 2个Conv3×3 |

**关键说明：**
1. **unetUp 模块**：由双线性上采样（2×）+ 2个 Conv3×3 + ReLU 组成
2. **跳跃连接**：将编码器对应层的特征图与解码器上采样后的特征图在通道维度拼接
3. **ResNet50 额外处理**：由于下采样倍数更大，需要额外的 `up_conv` 层将特征图从 256×256 上采样到 512×512
4. **最终输出**：通过 1×1 卷积将特征映射到 `num_classes` 个通道，每个通道对应一个类别的分割概率

### 关键文件

- `train.py`：标准 VOC 格式训练，带验证
- `train_medical.py`：医学数据集训练（无验证划分）
- `unet.py`：推理包装器，包含预测、FPS 测试、ONNX 转换
- `predict.py`：交互式预测脚本
- `get_miou.py`：mIoU 指标评估脚本
- `voc_annotation.py`：数据集划分生成和格式验证

### 数据流

1. **数据集格式**：期望 VOC 格式
   - 图像：`VOCdevkit/VOC2007/JPEGImages/*.jpg`
   - 标签：`VOCdevkit/VOC2007/SegmentationClass/*.png`（灰度图，像素值 = 类别 ID）
   - 划分文件：`VOCdevkit/VOC2007/ImageSets/Segmentation/*.txt`

2. **数据加载器**：
   - `utils/dataloader.py`：标准 VOC 数据集，带 train/val 划分
   - `utils/dataloader_medical.py`：医学数据集（仅训练集）

3. **训练过程**：
   - 通过列出图像 ID 的 txt 文件加载数据集
   - 训练期间应用数据增强
   - 使用 `utils/utils_fit.py` 进行训练轮次逻辑
   - `utils/callbacks.py` 中的回调用于损失历史记录和评估

## 重要配置要点

### 训练自定义数据集时

**训练文件中需要修改的关键参数：**
1. `num_classes`：类别数量 + 1（例如，二分类分割为 2+1）
2. `backbone`：选择 "vgg" 或 "resnet50"
3. `model_path`：预训练权重路径，或空字符串以从骨干网络预训练权重开始训练
4. `VOCdevkit_path`：数据集文件夹路径

**预测文件（`unet.py`）中：**
1. `model_path`：logs 文件夹中训练好的权重路径
2. `num_classes`：必须与训练配置匹配
3. `backbone`：必须与训练配置匹配

### 训练阶段

**冻结阶段**（`Freeze_Train = True`）：
- 骨干网络冻结，仅训练解码器
- 内存使用较低
- 防止早期训练中骨干网络权重损坏
- 默认：epoch 0-50，使用较小的学习率

**解冻阶段**：
- 全网络训练
- 内存使用较高
- 允许骨干网络适应特定任务
- 默认：epoch 50-100

### 损失函数配置

- `dice_loss`：少量类别时推荐设为 True，多类别且小批次时设为 False
- `focal_loss`：处理类别不平衡
- `cls_weights`：每个类别的损失权重（numpy 数组，长度 = num_classes）

### 多GPU训练

- Windows：自动使用 DP 模式（设置 `distributed = False`）
- Ubuntu：可以通过适当的命令行调用使用 DDP 模式
- `sync_bn`：在多GPU的 DDP 模式下使用同步批归一化
- `fp16`：混合精度训练（需要 PyTorch ≥1.7.1）

## 常见问题和重要说明

### 数据集格式要求

git@github.com:zdhgzw/bili-unet-pytorch.git




**标签图像必须**：
- 为 PNG 格式（不是 JPG）
- 为灰度图或 8 位彩色图像
- 像素值等于类别 ID（0 表示背景，1 表示类别 1，等等）
- **不能**在二分类分割中使用 0 和 255 的值（应使用 0 和 1）

**常见错误**：标签使用 background=0 和 target=255 可以训练但不会产生结果。必须使用 background=0 和 target=1。

### 形状不匹配错误

如果在预测期间遇到形状不匹配：
1. 检查训练和预测之间的 `num_classes` 是否匹配
2. 验证 `model_path` 是否指向正确的权重文件
3. 确保 `backbone` 与训练配置相同

### 内存问题

- 如果出现 OOM 错误，减少 `batch_size`
- 最小 `batch_size = 2`（由于 BatchNorm 层，不能为 1）
- 对于 ResNet50 骨干网络，`batch_size` 不能为 1
- 典型要求：
  - 2GB 显存：非常有限
  - 4GB 显存：使用小批次可行
  - 6GB+ 显存：训练舒适

### 预训练权重

- **始终使用预训练权重**（骨干网络或完整模型）
- 从随机初始化训练会产生较差的结果
- 骨干网络预训练权重与数据集无关（特征是通用的）
- 如果修改网络架构，可能需要自定义权重加载逻辑

### 模型路径配置

- `model_path = ''` + `pretrained = True`：仅加载骨干网络预训练权重
- `model_path = ''` + `pretrained = False` + `Freeze_Train = False`：从头训练（不推荐）
- `model_path = 'path/to/weights.pth'`：加载完整模型权重（用于微调或恢复训练）

## 文件组织结构

```
unet-pytorch/
├── nets/                    # 网络定义
│   ├── unet.py             # 主 U-Net 架构
│   ├── vgg.py              # VGG16 骨干网络
│   ├── resnet.py           # ResNet50 骨干网络
│   └── unet_training.py    # 训练工具（学习率调度器等）
├── utils/                   # 辅助函数
│   ├── dataloader.py       # VOC 数据集加载器
│   ├── dataloader_medical.py  # 医学数据集加载器
│   ├── utils_fit.py        # 训练轮次逻辑
│   ├── callbacks.py        # 训练回调
│   └── utils_metrics.py    # mIoU 计算
├── model_data/             # 预训练权重存储
├── logs/                   # 训练输出（权重、损失日志）
├── VOCdevkit/              # VOC 格式数据集
│   └── VOC2007/
│       ├── JPEGImages/     # 输入图像（.jpg）
│       ├── SegmentationClass/  # 标签掩码（.png）
│       └── ImageSets/Segmentation/  # train/val 划分文件
├── Medical_Datasets/       # 医学数据集（如果使用 train_medical.py）
├── train.py                # 主训练脚本（VOC 格式）
├── train_medical.py        # 医学数据集训练脚本
├── unet.py                 # 预测包装类
├── predict.py              # 交互式预测
├── get_miou.py            # 评估脚本
└── voc_annotation.py      # 数据集准备
```

## 训练建议

### 从预训练模型权重开始
- **Adam 优化器**：`Init_lr=1e-4`，`weight_decay=0`，`UnFreeze_Epoch=100`
- **SGD 优化器**：`Init_lr=1e-2`，`weight_decay=1e-4`，`UnFreeze_Epoch=100`

### 从骨干网络预训练权重开始
- **Adam 优化器**：`Init_lr=1e-4`，`weight_decay=0`，`UnFreeze_Epoch=100`
- **SGD 优化器**：`Init_lr=1e-2`，`weight_decay=1e-4`，`UnFreeze_Epoch=120-300`
- 需要更多轮次来逃离局部最优，因为骨干网络需要适应

### 批次大小
- 使用内存允许的最大批次大小
- 典型设置：`Freeze_batch_size = 1-2x Unfreeze_batch_size`
- 学习率会根据批次大小自动调整

## ONNX 导出

模型支持 ONNX 导出以进行部署。使用 `unet.py` 中 `Unet` 类的 `convert_to_onnx` 方法。
