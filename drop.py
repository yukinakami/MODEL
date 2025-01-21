import json

def clean_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用列表推导式删除 latest_value 字段中包含 '.mp4' 的条目
    cleaned_data = [entry for entry in data if '.mp4' not in (entry.get('image_path') or '')]

    # 输出清理后的数据
    print(f"Removed {len(data) - len(cleaned_data)} entries with '.mp4' in 'image_path'.")

    return cleaned_data

# 使用示例
json_file = "G://模型//data//news_data.json"
cleaned_data = clean_data(json_file)

# 保存清理后的数据到新的 JSON 文件
with open("G://模型//data//news_data.json", 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
