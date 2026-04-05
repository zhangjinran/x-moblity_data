# X-Mobility 模型三个主要函数的输出数据结构说明

本文档基于以下源代码文件分析得出：
- `model/x_mobility/x_mobility.py` – 主模型 `XMobility`
- `model/x_mobility/action_policy.py` – `ActionPolicy`, `MLPPolicy`
- `model/x_mobility/decoders.py` – `StyleGanDecoder`, `SegmentationHead`, `RgbHead`
- `model/x_mobility/rssm.py` – `RSSM`
- `model/x_mobility/encoders.py` – `ObservationEncoder`, `SpeedEncoder`, `ImageDINOEncoder`
- `model/x_mobility/diffusion_rgb.py` – `RGBDiffuser`
- `model/x_mobility/utils.py` – `pack_sequence_dim`, `unpack_sequence_dim`

配置文件（`config.gin`）中定义的关键参数：
- `ACTION_SIZE = 6`
- `FLAT_PATH_SIZE = 10` (5 个路径点，每点 2 维)
- `NUM_SEMANTIC_CLASSES = 10`
- `RSSM.hidden_state_dim = 1024`, `RSSM.state_dim = 512`
- `SpeedEncoder.out_channels = 32`
- `ImageDINOEncoder` → `out_channels = 768` (DINOv2‑small: hidden_size=384 ×2)
- `ActionPolicy.policy_state_dim = 2048`
- `SEQUENCE_LENGTH = 4` (训练时，推理时可变)

以下对三个函数的输入/输出进行规范说明。所有张量均使用 PyTorch 格式。

---

## 1. `inference` 方法

**文件位置**：`model/x_mobility/x_mobility.py`  
**装饰器**：`@torch.inference_mode()`  
**用途**：在线推理（滤波 + 决策），利用当前观测更新状态并输出动作、路径及可选解码结果。

### 输入参数
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `batch` | `Dict` | – | 必须包含 `image`, `history`, `sample`, `action`, `route` 等键（详见数据加载器） |
| `enable_semantic_inference` | `bool` | – | 是否返回语义分割，默认 `True` |
| `enable_rgb_inference` | `bool` | – | 是否返回 RGB 图像，默认 `False` |
| `enable_depth` | `bool` | – | 是否返回深度图，默认 `False` |

### 返回值
类型：`tuple`，包含 7 个元素。

| 索引 | 名称 | 类型 | 形状 | 值域 | 说明 |
|------|------|------|------|------|------|
| 0 | `action_output` | `torch.Tensor` 或 `None` | `(B, S, 6)` | `[-1, 1]` | 连续动作指令（油门/转向等） |
| 1 | `path_output` | `torch.Tensor` 或 `None` | `(B, S, 10)` | `[-1, 1]` | 未来 5 个路径点，交错存储 `[x1,y1,...,x5,y5]` |
| 2 | `history` | `torch.Tensor` | `(B, 1024)` | 实数 | RSSM 确定性隐藏状态 |
| 3 | `sample` | `torch.Tensor` | `(B, 512)` | 实数 | RSSM 随机状态样本 |
| 4 | `semantic_output` | `torch.Tensor` 或 `None` | `(B, S, 10, H, W)` | 逻辑值 | 语义分割 logits（10 类），当 `enable_semantic_inference=True` 且 `self.enable_semantic=True` 时存在 |
| 5 | `rgb_output` | `torch.Tensor` 或 `None` | `(B, S, 3, H, W)` | `[0, 1]` | RGB 重建图像，当 `enable_rgb_inference=True` 且相应解码器启用时存在 |
| 6 | `depth_output` | `torch.Tensor` 或 `None` | `(B, S, 1, H_d, W_d)` | 深度值（米） | 仅当 `enable_depth=True` 且使用 `ImageDepthAnythingEncoder` 时存在；当前配置下为 `None` |

**符号说明**：
- `B`：batch size
- `S`：输入序列长度（来自 `batch['image'].shape[1]`）
- `H, W`：输入图像的空间尺寸（与原始图像一致）
- `H_d, W_d`：深度编码器固定输入尺寸（如 `DEPTH_ANYTHING_IMAGE_SIZE`）

---

## 2. `forward` 方法

**文件位置**：`model/x_mobility/x_mobility.py`  
**用途**：训练前向传播，计算所有损失项所需的中间输出。

### 输入参数
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `batch` | `Dict` | – | 必须包含 `image`, `action`, `route_vectors`, `speed` 等键 |

### 返回值
类型：`Dict`，动态包含以下键（取决于配置开关）。假设所有开关开启（`enable_semantic=True`, `enable_rgb_stylegan=True`, `enable_rgb_diffusion=True`, `is_gwm_pretrain=False`）。

| 键名 | 形状 | 值域 | 说明 |
|------|------|------|------|
| `embedding` | `(B, S, 800)` | 实数 | 观测编码拼接（图像特征 768 + 速度特征 32） |
| `speed_features` | `(B, S, 32)` | 实数 | 速度编码输出 |
| `image_features` | `(B, S, 768)` | 实数 | DINOv2 图像特征 |
| `image_attentions` | `(B, S, H_p, W_p)` | 注意力权重 | 最后一层 CLS 对 patch 的注意力，`H_p=H_in//14, W_p=W_in//14` |
| `posterior` | `Dict` | – | 包含 `hidden_state` `(B, S, 1024)` 和 `sample` `(B, S, 512)` |
| `action` | `(B, S, 6)` | `[-1, 1]` | 动作指令（来自 `ActionPolicy`） |
| `path` | `(B, S, 10)` | `[-1, 1]` | 路径点（来自 `ActionPolicy`） |
| `semantic_segmentation_1` | `(B, S, 10, H, W)` | 逻辑值 | 语义分割（原分辨率），来自 `head_1` |
| `semantic_segmentation_2` | `(B, S, 10, H/2, W/2)` | 逻辑值 | 语义分割（1/2 分辨率），来自 `head_2` |
| `semantic_segmentation_4` | `(B, S, 10, H/4, W/4)` | 逻辑值 | 语义分割（1/4 分辨率），来自 `head_4` |
| `rgb_1` | `(B, S, 3, H, W)` | `[0, 1]` | RGB 重建（原分辨率），来自 `StyleGanDecoder` 或 `RGBDiffuser.inference` |
| `rgb_2` | `(B, S, 3, H/2, W/2)` | `[0, 1]` | RGB 重建（1/2 分辨率），来自 `StyleGanDecoder` |
| `rgb_4` | `(B, S, 3, H/4, W/4)` | `[0, 1]` | RGB 重建（1/4 分辨率），来自 `StyleGanDecoder` |
| `rgb_noise` | `(B*S, 4, H/8, W/8)` | 高斯噪声 | 扩散模型训练时采样的噪声，仅当 `self.training=True` 且启用 `RGBDiffuser` |
| `rgb_noise_pred` | `(B*S, 4, H/8, W/8)` | 预测噪声 | 扩散模型预测的噪声，仅当 `self.training=True` 且启用 `RGBDiffuser` |

**注意**：
- 当 `enable_rgb_diffusion=True` 且模型处于 `eval()` 模式时，`forward` 也会返回 `rgb_1`（通过调用 `self.rgb_diffuser.inference`）。
- `H/8, W/8` 是 VAE 编码后的 latent 空间尺寸（例如输入 `320×512` → `40×64`）。
- `B*S` 是经过 `pack_sequence_dim` 后的 batch 大小。

---

## 3. `inference_prediction` 方法

**文件位置**：`model/x_mobility/x_mobility.py`  
**装饰器**：`@torch.inference_mode()`  
**用途**：开环想象（纯预测），不依赖当前观测，仅基于历史状态和动作序列生成未来状态及解码输出。

### 输入参数
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `batch` | `Dict` | – | 必须包含 `image`, `history`, `sample`, `action` 等键 |
| `enable_semantic_inference` | `bool` | – | 默认 `True` |
| `enable_rgb_inference` | `bool` | – | 默认 `True` |

### 返回值
类型：`tuple`，包含 4 个元素。

| 索引 | 名称 | 类型 | 形状 | 值域 | 说明 |
|------|------|------|------|------|------|
| 0 | `history` | `torch.Tensor` | `(B, 1024)` | 实数 | 想象得到的确定性隐藏状态 |
| 1 | `sample` | `torch.Tensor` | `(B, 512)` | 实数 | 想象得到的随机状态样本 |
| 2 | `semantic_output` | `torch.Tensor` 或 `None` | `(B, 1, 10, H, W)` | 逻辑值 | 语义分割，时间维度固定为 1（单步预测） |
| 3 | `rgb_output` | `torch.Tensor` 或 `None` | `(B, 1, 3, H, W)` | `[0, 1]` | RGB 图像，时间维度固定为 1 |

**重要假设与约束**：
- 该函数内部调用 `self.rssm.imagine_step`，其输出为单步状态（无时间序列维度）。之后通过 `unpack_sequence_dim` 将形状 `(B, C, H, W)` 转换为 `(B, s, C, H, W)`，其中 `s = batch['image'].shape[1]`。
- 由于 `unpack_sequence_dim` 要求输入第一维等于 `B * s`，而实际输入第一维为 `B`，因此**仅在 `s = 1` 时函数正常工作**。在典型的单步预测场景中（例如模型预测控制），调用方应确保 `s = 1`。
- 若 `s > 1`，则形状不匹配，会引发运行时错误。因此，实际使用时 `semantic_output` 和 `rgb_output` 的形状为 `(B, 1, 10, H, W)` 和 `(B, 1, 3, H, W)`。

---

## 附录：关键常量与维度推导

| 符号 | 值 | 来源 |
|------|----|------|
| `ACTION_SIZE` | 6 | 配置文件 |
| `FLAT_PATH_SIZE` | 10 | 配置文件，对应 5 个 (x,y) 点 |
| `NUM_SEMANTIC_CLASSES` | 10 | 配置文件 |
| `hidden_state_dim` | 1024 | 配置文件 `RSSM.hidden_state_dim` |
| `state_dim` | 512 | 配置文件 `RSSM.state_dim` |
| `speed_encoder.out_channels` | 32 | 配置文件 `SpeedEncoder.out_channels` |
| `image_encoder.out_channels` | 768 | `ImageDINOEncoder` (DINOv2‑small: hidden_size=384 ×2) |
| `embedding_dim` | 800 | 768 + 32 |
| `policy_state_dim` | 2048 | 配置文件 `ActionPolicy.policy_state_dim` |
| `H, W` | 取决于数据集 | 输入图像尺寸（如 `320×512`） |
| `H/8, W/8` | VAE latent 尺寸 | 例如 `40×64` |
| `H_p, W_p` | `H_in//14, W_in//14` | DINOv2 patch 注意力图尺寸 |

以上为 X-Mobility 模型三个核心函数输出数据结构的完整说明。
