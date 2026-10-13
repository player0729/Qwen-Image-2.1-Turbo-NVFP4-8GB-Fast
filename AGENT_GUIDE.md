# Qwen-Image 2.1 Turbo 满血加速与零乱码定制方案 — 智能体执行手册 (AGENT_GUIDE)

> **致 AI 智能体 (Coding Agent)**：
> 本指南专为具备文件读写与终端命令执行能力的 AI 编程助手（如 Antigravity / Cursor / Claude Code / Windsurf 等）编写。
> 当用户要求你将本方案部署到他的 ComfyUI 环境中时，请严格按照本手册的 **【阶段一 ~ 阶段四】** 顺序执行，切勿猜测或遗漏步骤。

---

## 一、方案核心组件与 7 大底层改动概览

本方案针对 **ComfyUI (`v0.37.0+`)** 下的 **Qwen-Image 2.1 Turbo** 进行了全栈级协同优化，在 8GB 显存（如 RTX 5060 Laptop）上实现 **1K 4步 5s / 6步 8.8s、1.3K 12s、1.5K 16.8s、2K 28s~32s**，并彻底解决未指定具体文字时画面出现“一坨乱码字”的痛点：

1. **CK INT8 量化注意力机制 (`comfy kitchen attention`)**
   - **文件**：
     - `ComfyUI/comfy_extras/nodes_model_advanced.py` (`ModelAttentionBackend`)
     - `ComfyUI/comfy/ldm/modules/attention.py` (`attention_comfy_kitchen_int8`)
     - 工作流子图内部 `Node 486` (`ModelAttentionBackend` 已预置为 `['comfy kitchen attention']`)
   - **原理**：调用 `comfy_kitchen.int8_attention` / `prequantize_int8_attention` 执行硬件加速的 INT8 量化稠密注意力，替代繁琐的 SageAttention/Triton 编译依赖，在 Windows 下直接提供约 2 倍注意力计算提速。
2. **原位零额外显存 `Qwen3-VL-8B` 智能扩写补字 (`enable_prompt_enhance`)**
   - **文件**：`ComfyUI/comfy_extras/nodes_qwen.py` (`TextEncodeQwenImage21`)
   - **原理**：直接复用工作流已加载的 `qwen3vl_8b_w4a8_lmhead4bit.safetensors` 自带的 4-bit `lm_head`，通过 `<think>\n\n</think>\n\n` 免思考直出模式调用 `clip.generate()`，0 额外显存占用、无需额外部署 LLM 节点。内置针对 6 步蒸馏多峰坍缩定制的防乱码系统提示词（强制锁定主标题双引号、将分镜/招牌/字幕等抽象文字项展开为单行精炼双引号文案、禁止多余密集小字与换行碎片）。
3. **兼容 `PoseBranchCache` 的零显存驻留特征跳步缓存 (`enable_fbcache`)**
   - **文件**：`ComfyUI/comfy/ldm/qwen_image21/model.py` (`QwenImage21Transformer2DModel._forward`) & `ComfyUI/comfy_extras/nodes_qwen.py` (`QwenImage21Cache`)
   - **原理**：不走 `patches_replace`（避免触发 `hooked = True` 导致官方 INT8 Prefix KV 缓存失效），直接在 `_forward` 内部实现高噪期首块残差跳步（First-Block Cache）。仅在 `t > 0.91` 记录基线，2K 下基线自动卸载至 CPU RAM，跳步完成后立即销毁残差张量，实现 **0 MB 采样后显存驻留**，消除 2K 下 `WanVAE` 解码显存抖动（2K 稳定加速约 4 秒）。
4. **Qwen 2.1 原生训练桶分辨率选择器 (`ResolutionSelector` + `use_2k_sigmas` 自动联动)**
   - **文件**：`ComfyUI/comfy_extras/nodes_resolution.py`
   - **原理**：提供 `1K 极速档`、`1.3K 官方黄金桶`、`1.5K 海报排版档`、`2K 原生满血档` 四档 16 对齐官方训练桶，并输出第 3 个布尔端口 `use_2k_sigmas`（1K 时自动为 `False` 走 1K 锁字曲线，1.3K~2K 时自动为 `True` 走高分辨率曲线），直连外层主节点 `Node 459`。
5. **支持 4 步 / 6 步 / 8 步及任意步数的自适应锁字 Sigma 调度器**
   - **文件**：`ComfyUI/comfy_extras/nodes_custom_sampler.py` (`ManualSigmas`)
   - **原理**：为 1K 定制了 `0.328448` 锁字收尾锚点，为 1.3K/1.5K/2K 保持 `0.514892` 零错字蒸馏母曲线；同时将外层 `steps` 参数接入子图内所有 `ManualSigmas`，当用户在外层修改 `steps`（如改为 4 步）时，自动在定制母曲线上保持首尾与倒数第二步锁字锚点进行重采样，使 4 步、6 步、8 步均能真实生效。
6. **RTX 50 系 Blackwell (`sm_120`) 动态虚拟显存页表 (`comfy_aimdo`) 同步锁修复**
   - **文件**：`ComfyUI/comfy/model_patcher.py`、`ComfyUI/comfy/model_management.py` & `python/Lib/site-packages/comfy_aimdo/model_vbar.py`
   - **原理**：在 `vbar.prioritize()` 前插入 CUDA 流同步，在 `discard_cuda_async_error()` 中清理 `cudart` 异步错误码，并在 `vbar_fault` 返回错误码 `2` (`cuMemSetAccess: device not ready 600`) 时执行 `torch.cuda.synchronize` 重试与优雅回退，彻底根除连续生图时的 `RuntimeError: Fault failed: 2` 崩溃。
7. **外层主节点 (`Node 459`) 双开关 + 内部 CK 注意力全集成工作流**
   - **文件**：
     - `ComfyUI/user/default/workflows/qwen_image_2_1_image_turbo_6step_webp.json`
     - `ComfyUI/user/default/workflows/qwen_image_2_1_image_turbo.json`
   - **子图内置加速**：`Node 486` (`ModelAttentionBackend`) 默认启用 `comfy kitchen attention`，`Node 469` (`QwenImage21Cache`) 默认启用 `auto` + `int8` Prefix KV 缓存。
   - **外层面板底部双开关**：
     - `fbcache (跳步加速: 关=字准 / 开=快4s)`（默认 `false`）
     - `prompt_enhance (智能扩写补字: 关=原词 / 开=消除背景乱码字)`（默认 `false`）

---

## 二、阶段一：环境与依赖检查

在执行补丁安装前，请智能体先检查目标机器的 ComfyUI 环境与模型文件：

1. **目录定位**：找到包含 `ComfyUI/` 文件夹的根目录（例如 `D:\ComfyUI-aki\ComfyUI-aki-v3`）及其对应的 Python 解释器（例如 `<根目录>\python\python.exe`）。
2. **检查 `comfy_kitchen` (CK 注意力算子库) 是否已安装**：
   运行以下命令验证 CK INT8 注意力是否可用：
   ```powershell
   <Python路径> -c "import comfy_kitchen; print('CK available:', comfy_kitchen.int8_attention_is_available())"
   ```
   若未安装 `comfy_kitchen`，请运行 `<Python路径> -m pip install -U comfy-kitchen` 安装。
3. **核心模型文件清单**（位于 `<根目录>\ComfyUI\models\` 下）：
   - `diffusion_models/qwen_image_2.1_turbo_nvfp4_mixed.safetensors`（若非 RTX 50 系显卡，可在外层节点切换为对应的 INT8/FP8/GGUF 权重）
   - `text_encoders/qwen3vl_8b_w4a8_lmhead4bit.safetensors`（**关键**：必须为带 `lmhead` 的权重以支持原位智能扩写）
   - `vae/qwen_image_2.1_vae_bf16.safetensors`
   - `loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors`（可选，默认强度为 0）

---

## 三、阶段二：一键自动安装（推荐首选）

本打包目录已内置幂等安装、CK 算子检测与回滚脚本 `apply_patch_bundle.py`。

### 1. 预览检查（Dry-run）
```powershell
<Python路径> apply_patch_bundle.py --root "<ComfyUI整合包根目录>" --check
```

### 2. 执行备份与安装（自动备份原文件为 `.orig_backup` 并校验语法与 CK 算子）
```powershell
<Python路径> apply_patch_bundle.py --root "<ComfyUI整合包根目录>" --apply
```

### 3. 一键回滚（如需恢复原始文件）
```powershell
<Python路径> apply_patch_bundle.py --root "<ComfyUI整合包根目录>" --rollback
```

> **注意**：如果目标机器的 ComfyUI 版本为 `v0.37.0 / v0.37.1`，运行 `--apply` 即可 100% 完成全部源码与工作流部署。如果目标机器的 ComfyUI 版本经过大幅升级导致直接替换文件不兼容，请智能体改用下方 **【阶段三：源码级精准锚点移植】** 手动合并修改。

---

## 四、阶段三：源码级精准锚点移植（供智能体手动 Patch 使用）

若智能体需要在不同版本的 ComfyUI 上通过代码编辑工具手动合入补丁，请按以下精确锚点逐一修改（完整原文件均在 `files/` 目录下可供对照）：

### 补丁 1：`ComfyUI/comfy/model_patcher.py` & `ComfyUI/comfy/model_management.py`（修复 `vbar.prioritize()` 异步流竞态）
在 `ComfyUI/comfy/model_patcher.py` 的 `_load_dynamic_vram` 附近查找 `vbar.prioritize()`：
```python
# 修改前：
            if vbar is not None:
                vbar.prioritize()

# 修改后：
            if vbar is not None:
                comfy.model_management.synchronize()
                vbar.prioritize()
```

### 补丁 2：`python/Lib/site-packages/comfy_aimdo/model_vbar.py`（修复 `Fault failed: 2` 崩溃）
在 `ModelVBAR.fault` 方法中修改错误分支：
```python
# 修改后：
    def fault(self, alloc, size):
        offset = alloc - self.base_addr
        signature = (ctypes.c_uint32 * (size // (32 * 1024 ** 2) + 2))()
        res = lib.vbar_fault(self._devctx, self._ptr, offset, size, signature)
        if res == 0:
            return signature
        elif res == 1:
            return None
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            res = lib.vbar_fault(self._devctx, self._ptr, offset, size, signature)
            if res == 0:
                return signature
            return None
```

### 补丁 3：`ComfyUI/comfy/ldm/qwen_image21/model.py`（原生零显存驻留 `FBCache`）
1. 在 `QwenImage21Transformer2DModel.__init__` 末尾与 `reset_prefix_cache` 末尾添加状态字典：
```python
        self._fbcache_state = {"step": 0, "last_t": None, "consec_skips": 0, "d0": None, "d_rem": None}
```
2. 在 `QwenImage21Transformer2DModel._forward` 的 `self.transformer_blocks` 循环前后合入 FBCache 逻辑（完整实现直接参考 `files/ComfyUI/comfy/ldm/qwen_image21/model.py` 第 `335~434` 行）。

### 补丁 4：`ComfyUI/comfy_extras/nodes_qwen.py`（`enable_prompt_enhance` 与 `enable_fbcache`）
1. 在 `TextEncodeQwenImage21` 上方定义 `QWEN_IMAGE21_PE_SYSTEM_PROMPT`，并在 `TextEncodeQwenImage21.define_schema` 的 `inputs` 末尾添加 `enable_prompt_enhance` 布尔输入，在 `execute` 中加入原位 `clip.generate()` 扩写逻辑（完整代码见 `files/ComfyUI/comfy_extras/nodes_qwen.py` 第 `111~216` 行）。
2. 在 `QwenImage21Cache` 中添加 `enable_fbcache` 布尔输入，并将 `"fbcache_thresh": 0.45 if enable_fbcache else 0.0` 写入 `qwen_image21_cache` 字典。

### 补丁 5：`ComfyUI/comfy_extras/nodes_custom_sampler.py`（`ManualSigmas` 动态步数重采样）
在 `ManualSigmas` 节点中添加可选输入 `io.Int.Input("steps", default=0, min=0, max=10000, optional=True)`，当 `steps != len(sigmas) - 1` 时对非零段 `sigmas[:-1]` 线性插值重采样并保留首尾与倒数第二步锁字锚点（完整代码见 `files/ComfyUI/comfy_extras/nodes_custom_sampler.py` 第 `1125~1160` 行）。

### 补丁 6：`ComfyUI/comfy_extras/nodes_resolution.py`（Qwen 2.1 原生训练桶与 `use_2k_sigmas` 输出）
直接使用 `files/ComfyUI/comfy_extras/nodes_resolution.py` 替换同名文件，为外层 `Node 13 (ResolutionSelector)` 提供 `qwen_preset` 四档训练桶与 `use_2k_sigmas` 第三输出端口。

### 补丁 7：CK 注意力节点与后端 (`nodes_model_advanced.py` & `attention.py`)
若目标 ComfyUI 尚无 `ModelAttentionBackend` 节点，参考 `files/ComfyUI/comfy_extras/nodes_model_advanced.py` 第 `374~426` 行及 `files/ComfyUI/comfy/ldm/modules/attention.py` 合入 `ModelAttentionBackend` 与 `attention_comfy_kitchen_int8`。

---

## 五、阶段四：语法、CK 算子与工作流连线验收

部署完成后，智能体**必须**运行以下验收命令确认 0 报错：

```powershell
<Python路径> apply_patch_bundle.py --root "<ComfyUI整合包根目录>" --verify
```

确认输出 `ALL CHECKS PASSED` 后，通知用户**重启 ComfyUI 并重新加载工作流**即可。

---

## 六、外层主节点 (`Node 459`) 最佳实践指南

| 场景类型 | 推荐 `steps` | `prompt_enhance` (智能扩写补字) | `fbcache` (跳步加速) | 预期耗时 (RTX 5060 8G) |
| :--- | :---: | :---: | :---: | :---: |
| **分镜脚本 / 密集小字 / 海报菜单** | `6` | **`true` (开启)** | `false` (关闭) | 1K `~8.8s` / 2K `~32s` |
| **已写好精确长提示词的文字图** | `6` | `false` 或 `true` | `false` (关闭) | 1K `~8.8s` / 2K `~32s` |
| **纯风景 / 人像 / 无密集小字大图** | `6` | 按需开启 | **`true` (开启)** | 2K 稳定 `~28s` |
| **极速构图预览 / 仅含大标题海报** | `4` | 按需开启 | `false` (关闭) | 1K 仅 **`~5.0s`** |
