# Qwen-Image 2.1 Turbo 全面基准测试报告 (RTX 5060 Laptop 8GB)

- **测试显卡**：NVIDIA GeForce RTX 5060 Laptop GPU (8151 MB VRAM)
- **软件环境**：Windows 11 / Python 3.13.11 / PyTorch 2.9.1+cu130 / ComfyUI v0.37.1
- **注意力与前缀缓存**：`comfy kitchen attention` (INT8) + `QwenImage21Cache` (`auto`, `int8`)
- **测试方法**：每次切换模型或切换分辨率档位先执行 **1 次预热 (Warm-up)**；稳态测速采用 **`seed = 42, 43, 44` 连续运行 3 次取平均值 (Mean) ± 标准差 (Std)**；通过 `nvml.dll` 与 `psutil` 每 `50ms` 采样记录 **GPU 显存峰值 (`VRAM Peak`)** 与 **ComfyUI 进程内存峰值 (`Proc RAM Peak`)**；画质对比图统一固定 **`seed = 42`**。

## 一、模型切换与冷启动耗时 (Cold Start)

| 扩散模型 | 权重体积 | 首次加载/切换预热耗时 | 显存峰值 (VRAM Peak) | ComfyUI 内存峰值 (Proc RAM) | 系统总内存峰值 (Sys RAM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen-image-2.1-turbo-NVFP4.safetensors` (纯 NVFP4 (4.0GB)) | 4.00 GB | **12.15s** | 7533 MB (7.36 GB) | 4.48 GB | 14.89 GB |
| `qwen_image_2.1_turbo_nvfp4_mixed.safetensors` (混合 NVFP4 (5.4GB)) | 5.43 GB | **9.31s** | 7600 MB (7.42 GB) | 8.32 GB | 18.46 GB |

## 二、Part 1：模型精度 × 全分辨率档位 × 步数 性能与显存/内存基准表 (P1 数学高考试卷 `3:4`)

| 分辨率 (`3:4`) | 步数 | 混合 NVFP4 (均值 ± 标准差) | 纯 NVFP4 (均值 ± 标准差) | 纯 NVFP4 提速 | 混合 NVFP4 显存/内存峰值 | 纯 NVFP4 显存/内存峰值 | 混合 NVFP4 样张 | 纯 NVFP4 样张 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1K** | `4` 步 | **4.99s** ± 0.04s | **4.10s** ± 0.07s | **-0.89s (-17.8%)** | VRAM: 7479MB / RAM: 5.19G | VRAM: 6797MB / RAM: 4.50G | [样张](images/Part1_P1_Exam_3x4_1K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1K_4step_pureNVFP4.webp) |
| **1K** | `6` 步 | **6.58s** ± 0.17s | **5.33s** ± 0.05s | **-1.25s (-19.0%)** | VRAM: 7287MB / RAM: 5.20G | VRAM: 6797MB / RAM: 4.49G | [样张](images/Part1_P1_Exam_3x4_1K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1K_6step_pureNVFP4.webp) |
| **1K** | `8` 步 | **8.23s** ± 0.01s | **6.54s** ± 0.05s | **-1.69s (-20.5%)** | VRAM: 7319MB / RAM: 5.20G | VRAM: 6797MB / RAM: 4.50G | [样张](images/Part1_P1_Exam_3x4_1K_8step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1K_8step_pureNVFP4.webp) |
| **1.3K** | `4` 步 | **7.78s** ± 0.07s | **6.35s** ± 0.06s | **-1.42s (-18.3%)** | VRAM: 7479MB / RAM: 5.51G | VRAM: 7277MB / RAM: 4.53G | [样张](images/Part1_P1_Exam_3x4_1.3K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.3K_4step_pureNVFP4.webp) |
| **1.3K** | `6` 步 | **10.42s** ± 0.06s | **8.42s** ± 0.05s | **-2.00s (-19.2%)** | VRAM: 7415MB / RAM: 5.51G | VRAM: 7373MB / RAM: 4.53G | [样张](images/Part1_P1_Exam_3x4_1.3K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.3K_6step_pureNVFP4.webp) |
| **1.3K** | `8` 步 | **13.05s** ± 0.09s | **10.45s** ± 0.02s | **-2.61s (-20.0%)** | VRAM: 7607MB / RAM: 5.51G | VRAM: 7277MB / RAM: 4.53G | [样张](images/Part1_P1_Exam_3x4_1.3K_8step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.3K_8step_pureNVFP4.webp) |
| **1.5K** | `4` 步 | **8.27s** ± 0.23s | **6.77s** ± 0.07s | **-1.50s (-18.2%)** | VRAM: 7543MB / RAM: 5.89G | VRAM: 7021MB / RAM: 4.55G | [样张](images/Part1_P1_Exam_3x4_1.5K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.5K_4step_pureNVFP4.webp) |
| **1.5K** | `6` 步 | **11.23s** ± 0.01s | **9.02s** ± 0.11s | **-2.21s (-19.7%)** | VRAM: 7415MB / RAM: 5.89G | VRAM: 7021MB / RAM: 4.55G | [样张](images/Part1_P1_Exam_3x4_1.5K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.5K_6step_pureNVFP4.webp) |
| **1.5K** | `8` 步 | **14.11s** ± 0.04s | **11.33s** ± 0.01s | **-2.78s (-19.7%)** | VRAM: 7447MB / RAM: 5.89G | VRAM: 7021MB / RAM: 4.55G | [样张](images/Part1_P1_Exam_3x4_1.5K_8step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1.5K_8step_pureNVFP4.webp) |
| **2K** | `4` 步 | **22.86s** ± 0.10s | **19.65s** ± 0.09s | **-3.21s (-14.0%)** | VRAM: 8055MB / RAM: 9.13G | VRAM: 7501MB / RAM: 7.42G | [样张](images/Part1_P1_Exam_3x4_2K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_2K_4step_pureNVFP4.webp) |
| **2K** | `6` 步 | **31.52s** ± 0.17s | **26.73s** ± 0.20s | **-4.79s (-15.2%)** | VRAM: 7991MB / RAM: 9.13G | VRAM: 7341MB / RAM: 7.53G | [样张](images/Part1_P1_Exam_3x4_2K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_2K_6step_pureNVFP4.webp) |
| **2K** | `8` 步 | **40.28s** ± 0.13s | **34.05s** ± 0.04s | **-6.23s (-15.5%)** | VRAM: 8055MB / RAM: 9.29G | VRAM: 7341MB / RAM: 7.53G | [样张](images/Part1_P1_Exam_3x4_2K_8step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_2K_8step_pureNVFP4.webp) |

## 三、Part 2：全提示词 × 专属常用比例 (1K / 2K × 4步 / 6步) 对比表 (`seed=42`)

| 提示词场景 | 比例 | 分辨率 | 步数 | 混合 NVFP4 (耗时 / 显存 / 内存) | 纯 NVFP4 (耗时 / 显存 / 内存) | 官方原版参考图 | 混合 NVFP4 样张 | 纯 NVFP4 样张 |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **P1: 2026 数学高考试卷** | `3:4` | **1K** | `4` 步 | **4.99s** (7479MB / 5.19G) | **4.10s** (6797MB / 4.50G) | [官方原图](images/Official_Ref_P1_Exam.webp) | [样张](images/Part1_P1_Exam_3x4_1K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1K_4step_pureNVFP4.webp) |
| **P1: 2026 数学高考试卷** | `3:4` | **1K** | `6` 步 | **6.58s** (7287MB / 5.20G) | **5.33s** (6797MB / 4.49G) | [官方原图](images/Official_Ref_P1_Exam.webp) | [样张](images/Part1_P1_Exam_3x4_1K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_1K_6step_pureNVFP4.webp) |
| **P1: 2026 数学高考试卷** | `3:4` | **2K** | `4` 步 | **22.86s** (8055MB / 9.13G) | **19.65s** (7501MB / 7.42G) | [官方原图](images/Official_Ref_P1_Exam.webp) | [样张](images/Part1_P1_Exam_3x4_2K_4step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_2K_4step_pureNVFP4.webp) |
| **P1: 2026 数学高考试卷** | `3:4` | **2K** | `6` 步 | **31.52s** (7991MB / 9.13G) | **26.73s** (7341MB / 7.53G) | [官方原图](images/Official_Ref_P1_Exam.webp) | [样张](images/Part1_P1_Exam_3x4_2K_6step_mixedNVFP4.webp) | [样张](images/Part1_P1_Exam_3x4_2K_6step_pureNVFP4.webp) |
| **P2: 奥拉维城堡建筑图板** | `16:9` | **1K** | `4` 步 | **8.78s** (7831MB / 9.17G) | **8.37s** (7629MB / 7.43G) | [官方原图](images/Official_Ref_P2_Castle.webp) | [样张](images/Part2_P2_Castle_16x9_1K_4step_mixedNVFP4.webp) | [样张](images/Part2_P2_Castle_16x9_1K_4step_pureNVFP4.webp) |
| **P2: 奥拉维城堡建筑图板** | `16:9` | **1K** | `6` 步 | **6.63s** (7479MB / 9.14G) | **5.54s** (6669MB / 7.40G) | [官方原图](images/Official_Ref_P2_Castle.webp) | [样张](images/Part2_P2_Castle_16x9_1K_6step_mixedNVFP4.webp) | [样张](images/Part2_P2_Castle_16x9_1K_6step_pureNVFP4.webp) |
| **P2: 奥拉维城堡建筑图板** | `16:9` | **2K** | `4` 步 | **23.00s** (8023MB / 9.65G) | **20.04s** (8093MB / 7.70G) | [官方原图](images/Official_Ref_P2_Castle.webp) | [样张](images/Part2_P2_Castle_16x9_2K_4step_mixedNVFP4.webp) | [样张](images/Part2_P2_Castle_16x9_2K_4step_pureNVFP4.webp) |
| **P2: 奥拉维城堡建筑图板** | `16:9` | **2K** | `6` 步 | **31.77s** (7543MB / 9.65G) | **26.99s** (6861MB / 7.70G) | [官方原图](images/Official_Ref_P2_Castle.webp) | [样张](images/Part2_P2_Castle_16x9_2K_6step_mixedNVFP4.webp) | [样张](images/Part2_P2_Castle_16x9_2K_6step_pureNVFP4.webp) |
| **P3: Wanderlust 旅行 App 界面** | `9:16` | **1K** | `4` 步 | **8.20s** (7543MB / 9.55G) | **7.47s** (7757MB / 7.61G) | [官方原图](images/Official_Ref_P3_AppUI.webp) | [样张](images/Part2_P3_AppUI_9x16_1K_4step_mixedNVFP4.webp) | [样张](images/Part2_P3_AppUI_9x16_1K_4step_pureNVFP4.webp) |
| **P3: Wanderlust 旅行 App 界面** | `9:16` | **1K** | `6` 步 | **6.70s** (7351MB / 9.52G) | **5.34s** (6573MB / 7.57G) | [官方原图](images/Official_Ref_P3_AppUI.webp) | [样张](images/Part2_P3_AppUI_9x16_1K_6step_mixedNVFP4.webp) | [样张](images/Part2_P3_AppUI_9x16_1K_6step_pureNVFP4.webp) |
| **P3: Wanderlust 旅行 App 界面** | `9:16` | **2K** | `4` 步 | **22.35s** (7895MB / 9.64G) | **19.39s** (7821MB / 7.70G) | [官方原图](images/Official_Ref_P3_AppUI.webp) | [样张](images/Part2_P3_AppUI_9x16_2K_4step_mixedNVFP4.webp) | [样张](images/Part2_P3_AppUI_9x16_2K_4step_pureNVFP4.webp) |
| **P3: Wanderlust 旅行 App 界面** | `9:16` | **2K** | `6` 步 | **31.14s** (7319MB / 9.65G) | **26.20s** (7053MB / 7.70G) | [官方原图](images/Official_Ref_P3_AppUI.webp) | [样张](images/Part2_P3_AppUI_9x16_2K_6step_mixedNVFP4.webp) | [样张](images/Part2_P3_AppUI_9x16_2K_6step_pureNVFP4.webp) |
| **P4: 动漫婚纱角色立绘** | `3:4` | **1K** | `4` 步 | **7.29s** (7543MB / 9.55G) | **6.76s** (7501MB / 7.61G) | [官方原图](images/Official_Ref_P4_Bride.webp) | [样张](images/Part2_P4_Bride_3x4_1K_4step_mixedNVFP4.webp) | [样张](images/Part2_P4_Bride_3x4_1K_4step_pureNVFP4.webp) |
| **P4: 动漫婚纱角色立绘** | `3:4` | **1K** | `6` 步 | **6.54s** (7287MB / 9.51G) | **5.30s** (6957MB / 7.57G) | [官方原图](images/Official_Ref_P4_Bride.webp) | [样张](images/Part2_P4_Bride_3x4_1K_6step_mixedNVFP4.webp) | [样张](images/Part2_P4_Bride_3x4_1K_6step_pureNVFP4.webp) |
| **P4: 动漫婚纱角色立绘** | `3:4` | **2K** | `4` 步 | **22.77s** (7799MB / 9.64G) | **19.56s** (7789MB / 7.70G) | [官方原图](images/Official_Ref_P4_Bride.webp) | [样张](images/Part2_P4_Bride_3x4_2K_4step_mixedNVFP4.webp) | [样张](images/Part2_P4_Bride_3x4_2K_4step_pureNVFP4.webp) |
| **P4: 动漫婚纱角色立绘** | `3:4` | **2K** | `6` 步 | **31.53s** (7447MB / 9.64G) | **26.80s** (6893MB / 7.72G) | [官方原图](images/Official_Ref_P4_Bride.webp) | [样张](images/Part2_P4_Bride_3x4_2K_6step_mixedNVFP4.webp) | [样张](images/Part2_P4_Bride_3x4_2K_6step_pureNVFP4.webp) |
| **P5: 祈年殿拆解科普卡 (短提示词)** | `3:4` | **1K** | `4` 步 | **7.17s** (7575MB / 9.54G) | **6.38s** (7398MB / 7.60G) | [官方原图](images/Official_Ref_P5_Temple.webp) | [样张](images/Part2_P5_Temple_3x4_1K_4step_mixedNVFP4.webp) | [样张](images/Part2_P5_Temple_3x4_1K_4step_pureNVFP4.webp) |
| **P5: 祈年殿拆解科普卡 (短提示词)** | `3:4` | **1K** | `6` 步 | **6.41s** (7287MB / 9.49G) | **5.17s** (7053MB / 7.55G) | [官方原图](images/Official_Ref_P5_Temple.webp) | [样张](images/Part2_P5_Temple_3x4_1K_6step_mixedNVFP4.webp) | [样张](images/Part2_P5_Temple_3x4_1K_6step_pureNVFP4.webp) |
| **P5: 祈年殿拆解科普卡 (短提示词)** | `3:4` | **2K** | `4` 步 | **22.51s** (7415MB / 9.64G) | **19.20s** (7725MB / 7.69G) | [官方原图](images/Official_Ref_P5_Temple.webp) | [样张](images/Part2_P5_Temple_3x4_2K_4step_mixedNVFP4.webp) | [样张](images/Part2_P5_Temple_3x4_2K_4step_pureNVFP4.webp) |
| **P5: 祈年殿拆解科普卡 (短提示词)** | `3:4` | **2K** | `6` 步 | **31.26s** (7447MB / 9.63G) | **26.31s** (6541MB / 7.69G) | [官方原图](images/Official_Ref_P5_Temple.webp) | [样张](images/Part2_P5_Temple_3x4_2K_6step_mixedNVFP4.webp) | [样张](images/Part2_P5_Temple_3x4_2K_6step_pureNVFP4.webp) |
| **P6: 巨兽对决 3DCG 故事板 (短提示词)** | `1:1` | **1K** | `4` 步 | **7.21s** (7575MB / 9.54G) | **6.51s** (7437MB / 7.60G) | [官方原图](images/Official_Ref_P6_Monster.webp) | [样张](images/Part2_P6_Monster_1x1_1K_4step_mixedNVFP4.webp) | [样张](images/Part2_P6_Monster_1x1_1K_4step_pureNVFP4.webp) |
| **P6: 巨兽对决 3DCG 故事板 (短提示词)** | `1:1` | **1K** | `6` 步 | **6.36s** (7415MB / 9.50G) | **5.12s** (7149MB / 7.57G) | [官方原图](images/Official_Ref_P6_Monster.webp) | [样张](images/Part2_P6_Monster_1x1_1K_6step_mixedNVFP4.webp) | [样张](images/Part2_P6_Monster_1x1_1K_6step_pureNVFP4.webp) |
| **P6: 巨兽对决 3DCG 故事板 (短提示词)** | `1:1` | **2K** | `4` 步 | **21.79s** (7671MB / 9.64G) | **19.00s** (7949MB / 7.70G) | [官方原图](images/Official_Ref_P6_Monster.webp) | [样张](images/Part2_P6_Monster_1x1_2K_4step_mixedNVFP4.webp) | [样张](images/Part2_P6_Monster_1x1_2K_4step_pureNVFP4.webp) |
| **P6: 巨兽对决 3DCG 故事板 (短提示词)** | `1:1` | **2K** | `6` 步 | **30.15s** (7351MB / 9.63G) | **25.47s** (7309MB / 7.69G) | [官方原图](images/Official_Ref_P6_Monster.webp) | [样张](images/Part2_P6_Monster_1x1_2K_6step_mixedNVFP4.webp) | [样张](images/Part2_P6_Monster_1x1_2K_6step_pureNVFP4.webp) |

## 四、Part 3：外层开关 (`prompt_enhance` 扩写 & `fbcache` 缓存) 专项测试

| 测试专项 | 提示词与比例 | 分辨率与步数 | 开关状态 | 实测耗时 (s) | 显存峰值 (VRAM) | 进程内存峰值 (RAM) | 样张链接 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Prompt Enhance 扩写对比 | P5: 祈年殿拆解科普卡 (短提示词) (`3:4`) | 1.3K / `6`步 | `prompt_enhance=False` | **10.63s** | 7526 MB | 7.6 GB | [查看样张](images/Part3_P5_Temple_3x4_1.3K_6step_PE_Off_pureNVFP4.webp) |
| Prompt Enhance 扩写对比 | P5: 祈年殿拆解科普卡 (短提示词) (`3:4`) | 1.3K / `6`步 | `prompt_enhance=True` | **15.44s** | 7568 MB | 7.8 GB | [查看样张](images/Part3_P5_Temple_3x4_1.3K_6step_PE_On_pureNVFP4.webp) |
| Prompt Enhance 扩写对比 | P6: 巨兽对决 3DCG 故事板 (短提示词) (`1:1`) | 1.3K / `6`步 | `prompt_enhance=False` | **11.47s** | 7767 MB | 7.8 GB | [查看样张](images/Part3_P6_Monster_1x1_1.3K_6step_PE_Off_pureNVFP4.webp) |
| Prompt Enhance 扩写对比 | P6: 巨兽对决 3DCG 故事板 (短提示词) (`1:1`) | 1.3K / `6`步 | `prompt_enhance=True` | **16.35s** | 7639 MB | 7.82 GB | [查看样张](images/Part3_P6_Monster_1x1_1.3K_6step_PE_On_pureNVFP4.webp) |
| FBCache 一阶特征缓存对比 | P1: 2026 数学高考试卷 (`3:4`) | 2K / `6`步 | `fbcache=False` | **33.48s** | 8055 MB | 8.43 GB | [查看样张](images/Part3_P1_Exam_3x4_2K_6step_FBCache_Off_pureNVFP4.webp) |
| FBCache 一阶特征缓存对比 | P1: 2026 数学高考试卷 (`3:4`) | 2K / `6`步 | `fbcache=True` | **23.76s** | 7575 MB | 8.51 GB | [查看样张](images/Part3_P1_Exam_3x4_2K_6step_FBCache_On_pureNVFP4.webp) |
| FBCache 一阶特征缓存对比 | P2: 奥拉维城堡建筑图板 (`16:9`) | 2K / `6`步 | `fbcache=False` | **30.02s** | 7952 MB | 8.42 GB | [查看样张](images/Part3_P2_Castle_16x9_2K_6step_FBCache_Off_pureNVFP4.webp) |
| FBCache 一阶特征缓存对比 | P2: 奥拉维城堡建筑图板 (`16:9`) | 2K / `6`步 | `fbcache=True` | **23.73s** | 7191 MB | 8.53 GB | [查看样张](images/Part3_P2_Castle_16x9_2K_6step_FBCache_On_pureNVFP4.webp) |

## 五、实测结论与参数选型建议

1. **纯 NVFP4 (`4.0GB`) vs 混合精度 NVFP4 (`5.4GB`)**：
   - **速度与显存优势**：在 RTX 5060 Laptop (8GB) 上，纯 NVFP4 在全分辨率档位下比混合精度稳定快 **15% ~ 20%**（例如 `1K 4步` 从 `4.99s` 降至 **`4.10s`**，`1K 6步` 从 `6.58s` 降至 **`5.33s`**，`1.5K 6步` 从 `11.23s` 降至 **`9.02s`**，`2K 6步` 从 `31.52s` 降至 **`26.73s`**）。GPU 显存峰值降低约 `500~700 MB`，ComfyUI 进程内存占用降低约 `0.7~1.9 GB`，有效缓解了 2K 分辨率下逼近 8GB 显存上限时的动态分页开销。
   - **画质与小字差异**：两者在构图、光影、人物立绘、数学公式及大中号文字上表现基本一致；但在**极小字号或复杂笔画密集排布**时，混合精度 NVFP4 略稳于纯 NVFP4（例如在 `P2 奥拉维城堡` 2K 样张中，混合精度下的主桥小字 `① 主入口` 清晰可读，而纯 NVFP4 出现轻微笔画粘连；在 `P1 数学试卷` 1K/1.5K 样张中，混合精度的“每小题5分”错笔率略低）。

2. **步数选择（4 步 vs 6 步 vs 8 步）**：
   - **`4` 步**：适合无密集小字的人像立绘、动漫插画、大标题海报或快速构图预览（纯 NVFP4 下 `1K` 仅需 **`4.10s`**，`2K` 仅需 **`19.65s`**），但密集宋体小字会出现缺笔或局部粘连。
   - **`6` 步（推荐默认）**：Turbo 模型的文字收敛甜点，相比 4 步耗时仅增加 `1.2s`（1K）至 `7s`（2K），但正文小字与公式清晰度显著提升。
   - **`8` 步**：耗时比 6 步增加约 25%，文字改善幅度有限；与其使用 `1K 8步 (6.54s)`，不如直接使用 **`1.3K 6步 (8.42s)`** 或 **`1.5K 6步 (9.02s)`**，字迹清晰度提升远大于单纯加步数。

3. **分辨率档位建议（1K / 1.3K / 1.5K / 2K）**：
   - **日常无字/大字海报**：`1K 4步 (4.10s)` 或 `1K 6步 (5.33s)`。
   - **含中号/小号文字的卡片与分镜（性价比最高）**：**`1.3K 6步 (8.42s)`** 或 **`1.5K 6步 (9.02s)`** —— 纯 NVFP4 跑 1.5K 6 步仅需 9 秒，即可获得接近 2K 的文字可读性。
   - **超多模块展板/高考试卷/学术信息图**：**`2K 6步`**（纯 NVFP4 约 `26.7s`，若要求极限小字准确度可切回混合精度 NVFP4 约 `31.5s`）。

4. **外层开关（`prompt_enhance` 与 `fbcache`）**：
   - **`prompt_enhance`**：当输入**已逐字写明所有文本内容的长提示词**（如官方 P1~P3）时建议**关闭**；当输入**未列出具体标签文字的简短提示词**（如 `P5 祈年殿拆解科普卡`）时强烈建议**开启**——实测耗时仅增加约 `4.8s`（显存峰值几乎不变），即可将右侧引线原本 100% 乱码的伪文字转化为清晰的“柱础、檐角、斗拱、椽子、瓦当、脊兽”等中文标注。
   - **`fbcache`**：在 `2K 6步` 下开启可将耗时从 `30~33s` 缩短至 **`23.7s`**（提速约 21%~29%），显存峰值下降约 `500~760 MB`，适合非密集小字场景下的 2K 加速。

