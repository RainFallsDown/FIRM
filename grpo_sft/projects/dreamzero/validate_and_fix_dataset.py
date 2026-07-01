#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def validate_and_fix(dataset_path: str):
    dataset_path = Path(dataset_path)
    meta_dir = dataset_path / "meta"
    
    print(f"🔍 验证并修复数据集: {dataset_path}\n{'='*60}")
    
    errors_fixed = []
    
    # 1. 检查 info.json
    print("\n1️⃣ 检查 info.json...")
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    
    if "length" not in info:
        print("  ❌ 缺少 length 字段")
        return False
    if "features" not in info:
        print("  ❌ 缺少 features 字段")
        return False
    
    features = info["features"]
    print(f"  ✓ features: {len(features)} keys")
    print(f"  ✓ length: {info['length']}")
    
    # 2. 检查并修复 modality.json
    print("\n2️⃣ 检查 modality.json...")
    modality_path = meta_dir / "modality.json"
    with open(modality_path) as f:
        modality = json.load(f)
    
    fixed = False
    
    # 检查 video 模态格式
    if "video" in modality:
        for key, value in modality["video"].items():
            if isinstance(value, dict):
                print(f"  ❌ video.{key} 格式错误（是字典，应该是字符串）")
                if "original_key" in value:
                    modality["video"][key] = value["original_key"]
                    print(f"  ✅ 已修复: video.{key} = {value['original_key']}")
                    fixed = True
                    errors_fixed.append(f"video.{key} 格式")
    
    if fixed:
        # 备份并保存
        import shutil
        shutil.copy(modality_path, str(modality_path) + ".backup")
        with open(modality_path, 'w') as f:
            json.dump(modality, f, indent=2)
        print(f"\n  💾 已保存修复后的 modality.json")
    
    # 验证所有 original_key 是否在 features 中
    print("\n  验证 modality 键映射:")
    for mod_name, mod_data in modality.items():
        if mod_name in ["state", "action"]:
            for key, value in mod_data.items():
                if isinstance(value, dict) and "original_key" in value:
                    orig_key = value["original_key"]
                    if orig_key in features:
                        print(f"    ✓ {mod_name}.{key} -> {orig_key}")
                    else:
                        print(f"    ❌ {mod_name}.{key} -> {orig_key} (不在 features 中)")
        elif mod_name == "video":
            for key, value in mod_data.items():
                if isinstance(value, str):
                    if value in features:
                        print(f"    ✓ {mod_name}.{key} -> {value}")
                    else:
                        print(f"    ❌ {mod_name}.{key} -> {value} (不在 features 中)")
    
    # 3. 检查 episodes.jsonl
    print("\n3️⃣ 检查 episodes.jsonl...")
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]
    print(f"  ✓ {len(episodes)} episodes")
    
    # 4. 检查 stats.json
    print("\n4️⃣ 检查 stats.json...")
    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        has_quantiles = all("q01" in v and "q99" in v for v in stats.values())
        if has_quantiles:
            print(f"  ✓ stats.json 包含 q01/q99")
        else:
            print(f"  ⚠️  stats.json 缺少 q01/q99")
    else:
        print(f"  ⚠️  stats.json 不存在")
    
    # 总结
    print(f"\n{'='*60}")
    if errors_fixed:
        print(f"✅ 已修复 {len(errors_fixed)} 个错误:")
        for err in errors_fixed:
            print(f"  - {err}")
    else:
        print("✅ 数据集格式正确，无需修复")
    
    return True

if __name__ == "__main__":
    dataset_path = "${TIANQING_DATA_ROOT:-/path/to/tianqing}/tianqing_data/data_valid/A2p_dataset_0302_330"
    validate_and_fix(dataset_path)
