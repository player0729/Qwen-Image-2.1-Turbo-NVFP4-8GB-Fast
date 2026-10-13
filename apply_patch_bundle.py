#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image 2.1 Turbo 满血加速与零乱码定制方案 — 自动安装 / 校验 / 回滚脚本
用法:
  python apply_patch_bundle.py --root "D:\\ComfyUI-aki\\ComfyUI-aki-v3" --check
  python apply_patch_bundle.py --root "D:\\ComfyUI-aki\\ComfyUI-aki-v3" --apply
  python apply_patch_bundle.py --root "D:\\ComfyUI-aki\\ComfyUI-aki-v3" --verify
  python apply_patch_bundle.py --root "D:\\ComfyUI-aki\\ComfyUI-aki-v3" --rollback
"""

import argparse
import hashlib
import json
import os
import py_compile
import shutil
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BUNDLE_DIR, "files")

RELATIVE_FILES = [
    os.path.join("ComfyUI", "comfy", "model_patcher.py"),
    os.path.join("ComfyUI", "comfy", "model_management.py"),
    os.path.join("python", "Lib", "site-packages", "comfy_aimdo", "model_vbar.py"),
    os.path.join("ComfyUI", "comfy", "ldm", "qwen_image21", "model.py"),
    os.path.join("ComfyUI", "comfy", "ldm", "modules", "attention.py"),
    os.path.join("ComfyUI", "comfy_extras", "nodes_model_advanced.py"),
    os.path.join("ComfyUI", "comfy_extras", "nodes_qwen.py"),
    os.path.join("ComfyUI", "comfy_extras", "nodes_custom_sampler.py"),
    os.path.join("ComfyUI", "comfy_extras", "nodes_resolution.py"),
    os.path.join("ComfyUI", "user", "default", "workflows", "qwen_image_2_1_image_turbo_6step_webp.json"),
    os.path.join("ComfyUI", "user", "default", "workflows", "qwen_image_2_1_image_turbo.json"),
]


def sha256_short(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def check_ck_attention():
    try:
        import comfy_kitchen
        has_int8_sdpa = bool(getattr(comfy_kitchen, "int8_attention_is_available", lambda: False)())
        print(f"  [OK] comfy_kitchen 已安装 (INT8 Attention 可用={has_int8_sdpa})")
    except ImportError:
        print("  [WARN] 当前 Python 环境未安装 comfy_kitchen，请运行: python -m pip install -U comfy-kitchen")


def check_env(root_dir):
    print(f"[Check] 目标根目录: {root_dir}")
    check_ck_attention()
    ok = True
    for rel in RELATIVE_FILES:
        src = os.path.join(FILES_DIR, rel)
        dst = os.path.join(root_dir, rel)
        src_h = sha256_short(src)
        dst_h = sha256_short(dst)
        status = "IDENTICAL (已最新)" if src_h == dst_h else ("READY TO UPDATE" if os.path.exists(dst) else "NEW FILE")
        print(f"  - {rel}\n      Bundle={src_h} | Target={dst_h} -> [{status}]")
        if src_h == "MISSING":
            ok = False
    return ok


def verify_bundle(root_dir):
    print(f"[Verify] 正在校验目标环境语法、CK 注意力算子与工作流连线: {root_dir}")
    check_ck_attention()
    for rel in RELATIVE_FILES:
        dst = os.path.join(root_dir, rel)
        if not os.path.exists(dst):
            if "comfy_aimdo" in rel:
                print(f"  [Skip] 未检测到独立 python/Lib/site-packages/comfy_aimdo: {rel}")
                continue
            raise FileNotFoundError(f"缺少目标文件: {dst}")
        if dst.endswith(".py"):
            py_compile.compile(dst, doraise=True)
            print(f"  [OK] py_compile 通过: {rel}")
        elif dst.endswith(".json"):
            with open(dst, "r", encoding="utf-8") as f:
                wf = json.load(f)
            sg = wf["definitions"]["subgraphs"][0]
            nodes_by_id = {n["id"]: n for n in sg["nodes"]}
            ck_nodes = [n for n in sg["nodes"] if n.get("type") == "ModelAttentionBackend"]
            assert ck_nodes and ck_nodes[0]["widgets_values"] == ["comfy kitchen attention"], "CK Attention node missing!"
            for idx, sgin in enumerate(sg["inputs"]):
                for lid in sgin["linkIds"]:
                    lk = [l for l in sg["links"] if l["id"] == lid][0]
                    tn = nodes_by_id[lk["target_id"]]
                    ts = tn["inputs"][lk["target_slot"]]
                    assert lk["origin_slot"] == idx, f"origin_slot mismatch in {rel}: {lk} vs {idx}"
                    assert ts["link"] == lid, f"target_slot link mismatch in {rel}: {ts} vs {lid}"
            print(f"  [OK] 子图连线与 CK 注意力节点校验通过 ({len(sg['inputs'])} inputs): {rel}")
    print("ALL CHECKS PASSED")


def apply_bundle(root_dir):
    print(f"[Apply] 正在备份并部署定制文件到: {root_dir}")
    for rel in RELATIVE_FILES:
        src = os.path.join(FILES_DIR, rel)
        dst = os.path.join(root_dir, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"打包源文件缺失: {src}")
        if "comfy_aimdo" in rel and not os.path.exists(os.path.dirname(dst)):
            print(f"  [Skip] 目标环境不含本地 python/Lib/site-packages/comfy_aimdo，跳过 {rel}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            orig_bak = dst + ".orig_backup"
            if not os.path.exists(orig_bak):
                shutil.copy2(dst, orig_bak)
                print(f"  [Backup] 已创建原始备份: {rel}.orig_backup")
        shutil.copy2(src, dst)
        print(f"  [Installed] 已部署: {rel} (sha256={sha256_short(dst)})")
    verify_bundle(root_dir)


def rollback_bundle(root_dir):
    print(f"[Rollback] 正在从 .orig_backup 恢复原始文件: {root_dir}")
    for rel in RELATIVE_FILES:
        dst = os.path.join(root_dir, rel)
        orig_bak = dst + ".orig_backup"
        if os.path.exists(orig_bak):
            shutil.copy2(orig_bak, dst)
            print(f"  [Restored] 已恢复: {rel}")
        else:
            print(f"  [Skip] 未找到备份文件: {rel}.orig_backup")


def main():
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 Turbo 定制方案安装与校验工具")
    parser.add_argument("--root", type=str, default=os.path.abspath(os.path.join(BUNDLE_DIR, "..")),
                        help="ComfyUI 整合包根目录（包含 ComfyUI 文件夹的目录）")
    parser.add_argument("--check", action="store_true", help="对比打包文件与目标目录文件状态")
    parser.add_argument("--apply", action="store_true", help="自动备份并安装全部定制源码与工作流")
    parser.add_argument("--verify", action="store_true", help="仅运行 py_compile 与工作流连线校验")
    parser.add_argument("--rollback", action="store_true", help="回滚恢复到安装前的 .orig_backup")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.exists(os.path.join(root_dir, "ComfyUI")):
        print(f"[Error] 在 {root_dir} 下未找到 ComfyUI 目录，请通过 --root 指定正确的整合包根目录。")
        sys.exit(1)

    if args.rollback:
        rollback_bundle(root_dir)
    elif args.apply:
        apply_bundle(root_dir)
    elif args.verify:
        verify_bundle(root_dir)
    else:
        check_env(root_dir)


if __name__ == "__main__":
    main()
