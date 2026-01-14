import joblib
import pickle
import os
import argparse
import sys


def extract_core_data(source_dict):
    """辅助函数：从一个包含大量杂项的字典里提取核心数据"""
    keys_to_keep = ['pose_world', 'trans_world', 'betas', 'frame_ids']
    clean_dict = {}
    found = False
    for k in keys_to_keep:
        if k in source_dict:
            clean_dict[k] = source_dict[k]
            found = True
    return clean_dict if found else None


def process_file(input_path, output_path):
    print(f"正在加载: {input_path} ...")

    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件 '{input_path}'")
        return

    try:
        # 1. 加载数据 (兼容 joblib)
        raw_data = joblib.load(input_path)
        print("✅ 加载成功！")

        final_data = {}
        data_found = False

        # 2. 智能结构分析
        # 检查第一层是不是字典
        if isinstance(raw_data, dict):
            keys = list(raw_data.keys())
            if not keys:
                print("❌ 字典为空。")
                return

            first_val = raw_data[keys[0]]

            # 判断依据：如果里面的值还是字典，说明是嵌套的多人结构 {0: {...}, 1: {...}}
            if isinstance(first_val, dict) and ('pose_world' in first_val or 'pose' in first_val):
                print(f"🕵️  检测到【多人/嵌套】结构。包含 ID: {keys}")

                # 遍历每个人进行清洗
                for pid, person_data in raw_data.items():
                    # print(f"  -> 处理 Person ID: {pid}") # 减少刷屏，可注释回来
                    cleaned = extract_core_data(person_data)
                    if cleaned:
                        final_data[pid] = cleaned
                        data_found = True
                    else:
                        print(f"     ⚠️ 警告: Person {pid} 中没找到 pose/trans 数据，跳过。")

            # 判断依据：如果直接包含 pose_world，说明是单人扁平结构
            elif 'pose_world' in raw_data:
                print("🕵️  检测到【单人/扁平】结构。")
                cleaned = extract_core_data(raw_data)
                if cleaned:
                    final_data = cleaned
                    data_found = True

            else:
                print("❌ 未知结构：既不是标准的多人嵌套，也不是单人扁平。")
                print("Keys:", keys[:5])  # 只打印前几个key
                # 这种情况下，尝试保存原样，或者需要人工检查

        elif isinstance(raw_data, list):
            print(f"🕵️  检测到【列表】结构，包含 {len(raw_data)} 人。")
            # 如果是列表，通常也是多人
            final_data = []  # 保持列表结构
            for i, person_data in enumerate(raw_data):
                # print(f"  -> 处理 List Index: {i}")
                cleaned = extract_core_data(person_data)
                if cleaned:
                    final_data.append(cleaned)
            if final_data:
                data_found = True

        # 3. 保存结果
        if data_found and final_data:
            # 确保输出目录存在
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            with open(output_path, 'wb') as f:
                pickle.dump(final_data, f)

            # 大小对比
            old_size = os.path.getsize(input_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n🎉 瘦身完成！")
            print(f"📂 原文件: {old_size:.2f} MB")
            print(f"💾 新文件: {new_size:.2f} MB")
            print(f"📉 压缩率: {(1 - new_size / old_size) * 100:.1f}%")
            print(f"🚀 输出已保存至: {output_path}")
        else:
            print("\n❌ 提取失败，有效数据为空。")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    # 配置命令行参数
    parser = argparse.ArgumentParser(description="PKL 文件核心数据提取/瘦身工具")

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入 .pkl 文件路径 (例如: raw.pkl)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出 .pkl 文件路径 (例如: wushu2.pkl)'
    )

    args = parser.parse_args()

    process_file(args.input, args.output)