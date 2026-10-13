import math
from enum import Enum
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_V = "2:3 (Portrait Photo)"
    PHOTO_H = "3:2 (Photo)"
    STANDARD_V = "3:4 (Portrait Standard)"
    STANDARD_H = "4:3 (Standard)"
    WIDESCREEN_V = "9:16 (Portrait Widescreen)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"


class QwenPreset(str, Enum):
    TIER_1K = "1K 极速档 (1024级 · 日常与文字秒出)"
    TIER_1_3K = "1.3K 官方黄金桶 (1328级 · 通义原生预设)"
    TIER_1_5K = "1.5K 海报排版档 (1536级 · 严格网格对齐)"
    TIER_2K = "2K 原生满血档 (2048级 · 极致细节与微缩字)"


ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}

# Exact native training buckets for Qwen-Image / Qwen-Image-2.1 (all divisible by 16)
QWEN_BUCKETS: dict[str, dict[str, tuple[int, int]]] = {
    QwenPreset.TIER_1K: {
        "1:1": (1024, 1024),
        "2:3": (832, 1248),
        "3:2": (1248, 832),
        "3:4": (896, 1184),
        "4:3": (1184, 896),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
        "21:9": (1536, 640),
    },
    QwenPreset.TIER_1_3K: {
        "1:1": (1328, 1328),
        "2:3": (1056, 1584),
        "3:2": (1584, 1056),
        "3:4": (1104, 1472),
        "4:3": (1472, 1104),
        "9:16": (928, 1664),
        "16:9": (1664, 928),
        "21:9": (1920, 832),
    },
    QwenPreset.TIER_1_5K: {
        "1:1": (1536, 1536),
        "2:3": (1024, 1536),
        "3:2": (1536, 1024),
        "3:4": (1152, 1536),
        "4:3": (1536, 1152),
        "9:16": (1024, 1792),
        "16:9": (1792, 1024),
        "21:9": (2048, 896),
    },
    QwenPreset.TIER_2K: {
        "1:1": (2048, 2048),
        "2:3": (1696, 2528),
        "3:2": (2528, 1696),
        "3:4": (1792, 2400),
        "4:3": (2400, 1792),
        "9:16": (1536, 2752),
        "16:9": (2752, 1536),
        "21:9": (2944, 1280),
    },
}


def _extract_ratio_key(aspect_ratio: str) -> str:
    raw = aspect_ratio.value if isinstance(aspect_ratio, Enum) else str(aspect_ratio)
    prefix = raw.split(" ", 1)[0].strip()
    return prefix if prefix in ASPECT_RATIOS else "1:1"


class ResolutionSelector(io.ComfyNode):
    """Select exact Qwen-Image-2.1 native training bucket resolutions."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ResolutionSelector",
            display_name="Resolution Selector (Qwen 2.1 专属适配)",
            category="utilities",
            description="Select exact Qwen-Image-2.1 native training bucket resolutions (1K / 1.3K / 1.5K / 2K).",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.STANDARD_V,
                    tooltip="选择画面宽高比。",
                ),
                io.Combo.Input(
                    "qwen_preset",
                    options=QwenPreset,
                    default=QwenPreset.TIER_1K,
                    tooltip="Qwen-Image-2.1 专属原生训练桶档位。直接输出官方黄金训练尺寸，避免错字与位置编码偏移。",
                ),
                io.ResolutionPreview.Input(
                    "preview",
                    tooltip="当前输出分辨率与百万像素实时预览。",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "width", tooltip="输出宽度（像素）。"
                ),
                io.Int.Output(
                    "height", tooltip="输出高度（像素）。"
                ),
                io.Boolean.Output(
                    "use_2k_sigmas",
                    display_name="use_2k_sigmas",
                    tooltip="仅在 1K 极速档时为 False（关闭），在 1.3K / 1.5K / 2K 档位时自动为 True（开启），可直接连入采样器的 use_2k_sigmas。",
                ),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, qwen_preset: str = QwenPreset.TIER_1K, preview=None, **kwargs) -> io.NodeOutput:
        ratio_key = _extract_ratio_key(aspect_ratio)
        preset_key = qwen_preset.value if isinstance(qwen_preset, Enum) else str(qwen_preset)
        buckets = QWEN_BUCKETS.get(preset_key, QWEN_BUCKETS[QwenPreset.TIER_1K])
        width, height = buckets.get(ratio_key, (1024, 1024))
        use_2k_sigmas = not preset_key.startswith("1K")
        return io.NodeOutput(width, height, use_2k_sigmas)


class ResolutionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ResolutionSelector,
        ]


async def comfy_entrypoint() -> ResolutionExtension:
    return ResolutionExtension()

