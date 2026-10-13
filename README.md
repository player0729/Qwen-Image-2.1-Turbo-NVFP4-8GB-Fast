# QwenImage2.1-1k-5s-2k-28s-5060-8g

在 RTX 5060 Laptop (8GB VRAM) 笔记本上运行 ComfyUI `Qwen-Image 2.1 Turbo` 的配置、代码修改记录与工作流备份。

---

## 1. 硬件与模型配置参数

### 硬件与软件环境
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- **OS**: Windows 11
- **Python / PyTorch**: Python 3.13.11 / PyTorch 2.9.1+cu130
- **ComfyUI**: v0.37.1（启用 `comfy-kitchen` 与 `comfy_aimdo` 动态显存管理）

### 使用的模型文件
- **扩散模型 (`diffusion_models`)**: `qwen_image_2.1_turbo_nvfp4_mixed.safetensors`
- **文本编码器 (`text_encoders`)**: `qwen3vl_8b_w4a8_lmhead4bit.safetensors`
- **VAE (`vae`)**: `qwen_image_2.1_vae_bf16.safetensors`
- **采样参数**: `euler` + `simple`，`CFG = 1.0`，`4 / 6 / 8` 步

---

## 2. 我们做了什么

1. **启用 Comfy Kitchen INT8 注意力与 Prefix KV 缓存**
   - 在工作流子图中接入 `ModelAttentionBackend`（选择 `comfy kitchen attention`）与 `QwenImage21Cache`（`device=auto`, `dtype=int8`），缓存文本与参考图的前缀 KV，减少每步重复计算。

2. **分分辨率调整 Sigma 调度曲线，并支持 4/6/8 步动态重采样**
   - `ComfyUI/comfy_extras/nodes_custom_sampler.py` & `nodes_resolution.py`：
   - 针对 1K 分辨率调整了 6 步 Sigma 曲线（`1.0, 0.934262, 0.848546, 0.741433, 0.584844, 0.328448, 0.0`），降低小字漏笔率；1.3K / 1.5K / 2K 保持标准蒸馏曲线（`1.0, 0.972236, 0.937186, 0.89508, 0.805499, 0.514892, 0.0`）。
   - 为 `ManualSigmas` 增加 `steps` 输入并连接外层面板步数，修改外层 `steps`（如改为 4 步）时自动在当前曲线上插值重采样。
   - 修改 `ResolutionSelector` 节点，提供 1K / 1.3K / 1.5K / 2K 原生 16 对齐尺寸预设，并自动输出 `use_2k_sigmas` 布尔信号切换调度曲线。

3. **复用已加载的 `Qwen3-VL-8B` 做提示词扩写补字 (`prompt_enhance` 开关)**
   - `ComfyUI/comfy_extras/nodes_qwen.py` (`TextEncodeQwenImage21`)：
   - 在 `CFG=1.0` 的 6 步蒸馏模型中，如果提示词包含招牌、分镜、包装等载体但未写出具体文字，画面容易生成乱码。
   - 我们在 `TextEncodeQwenImage21` 内部直接调用已加载的 `qwen3vl_8b_w4a8_lmhead4bit`（利用其自带的 4-bit `lm_head`）在编码前生成具体文字描述并用双引号标出，无需额外加载第二个 LLM 模型，不增加额外显存占用。该功能作为布尔开关放在工作流外层节点上，可按需开启或关闭。

4. **内置首块残差跳步缓存 (`fbcache` 开关)**
   - `ComfyUI/comfy/ldm/qwen_image21/model.py` & `ComfyUI/comfy_extras/nodes_qwen.py` (`QwenImage21Cache`)：
   - 在 `QwenImage21Transformer2DModel._forward` 内部增加了可选的 First-Block Residual Cache（避免使用 `patches_replace` 导致 `QwenImage21Cache` 失效）。
   - 仅在高噪期（`t > 0.91`）记录残差，2K 分辨率下将基线卸载至内存，并在跳步完成后立即释放张量，避免占用 `WanVAE` 解码显存。同样作为外层开关提供（默认关闭）。

5. **修复 RTX 50 系显卡 `Fault failed: 2` 报错**
   - `ComfyUI/comfy/model_patcher.py`、`ComfyUI/comfy/model_management.py`、`python/Lib/site-packages/comfy_aimdo/model_vbar.py`：
   - 在 `vbar.prioritize()` 前及 `vbar_fault` 异常分支增加 CUDA 流同步与重试，解决连续生成时偶发的 `RuntimeError: Fault failed: 2`（`cuMemSetAccess: device not ready`）。

---

## 3. 实测效果与耗时

以下为 RTX 5060 Laptop (8GB) 上的实测生成耗时（不含首次冷启动加载模型时间）：

| 分辨率 | 步数 (`steps`) | `prompt_enhance` | `fbcache` | 实测耗时 | 实测效果说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1K (1024×1024)** | 4 | 关 | 关 | **~5.0s** | 大字可用，密集小字较糊，适合快速预览构图 |
| **1K (1024×1024)** | 6 | 关 / 开 | 关 | **~8.8s** | 错字明显减少，日常主力设置 |
| **1.3K (1328×1328)** | 6 | 关 / 开 | 关 | **~12.0s** | 画面与文字较稳定 |
| **1.5K (1536×1536)** | 6 | 关 / 开 | 关 | **~16.8s** | 适合海报与多格排版 |
| **2K (2048×2048)** | 6 | 关 | **开** | **~28.2s** | 跳过 1 步，耗时减少约 4s，但密集小字错字率会上升，适合无字/少字图 |
| **2K (2048×2048)** | 6 | 关 / **开** | 关 | **~32.0s** | 文字准确率高；开启 `prompt_enhance`（约增加 1~2s 扩写时间）可解决未指定具体文字时的背景乱码，并改善部分难字出字率 |

---

## 4. 安装与回滚方法

### 方式一：让 AI 智能体按说明自动部署
将本目录交给你的 AI 编程助手，让其阅读 [`AGENT_GUIDE.md`](AGENT_GUIDE.md) 并执行部署与校验。

### 方式二：命令行脚本安装
```bash
# 检查将要修改的文件
python apply_patch_bundle.py --root "你的ComfyUI整合包根目录" --check

# 自动备份原文件为 .orig_backup 并覆盖安装
python apply_patch_bundle.py --root "你的ComfyUI整合包根目录" --apply

# 恢复原始备份文件
python apply_patch_bundle.py --root "你的ComfyUI整合包根目录" --rollback
```

## License
[MIT](LICENSE)
