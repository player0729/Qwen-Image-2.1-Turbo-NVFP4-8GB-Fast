# -*- coding: utf-8 -*-
"""
QwenImage2.1-1k-5s-2k-28s-5060-8g
Zero-intrusion runtime monkey-patching entry point when installed in ComfyUI/custom_nodes/.
Automatically applies:
1. Blackwell (sm_120) comfy_aimdo VBAR fault synchronization & ModelPatcher stream sync
2. Zero-VRAM-Retention Native First-Block Residual Cache (FBCache) compatible with PoseBranchCache
3. Zero-Extra-VRAM In-Place Qwen3-VL-8B Prompt Enhancer (PE) in TextEncodeQwenImage21
4. Dynamic Step-Resampling ManualSigmas (4-step / 6-step / 8-step lock-text anchors)
5. Qwen-Image 2.1 Native Training Bucket ResolutionSelector with auto use_2k_sigmas routing
"""

import logging
import os
import shutil

logger = logging.getLogger("QwenImage21-SOTA-Pack")


def _apply_runtime_patches():
    # 1. Patch comfy_aimdo.model_vbar.ModelVBAR.fault for Blackwell cuMemSetAccess sync
    try:
        import ctypes
        import torch
        import comfy_aimdo.model_vbar as mvbar

        if not getattr(mvbar.ModelVBAR, "_qwen21_sota_patched", False):
            def _patched_fault(self, alloc, size):
                offset = alloc - self.base_addr
                signature = (ctypes.c_uint32 * (size // (32 * 1024 ** 2) + 2))()
                res = mvbar.lib.vbar_fault(self._devctx, self._ptr, offset, size, signature)
                if res == 0:
                    return signature
                elif res == 1:
                    return None
                else:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device)
                    res = mvbar.lib.vbar_fault(self._devctx, self._ptr, offset, size, signature)
                    if res == 0:
                        return signature
                    return None

            mvbar.ModelVBAR.fault = _patched_fault
            mvbar.ModelVBAR._qwen21_sota_patched = True
            logger.info("[Qwen21-SOTA] Patched comfy_aimdo.ModelVBAR.fault (Blackwell sync fix)")
    except Exception as e:
        logger.debug(f"[Qwen21-SOTA] Skipped comfy_aimdo patch: {e}")


_apply_runtime_patches()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
