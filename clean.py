import json

def clean_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用列表推导式删除 latest_value 为 None 的条目
    cleaned_data = [entry for entry in data if entry.get('latest_value') is not None]

    # 输出清理后的数据
    print(f"Removed {len(data) - len(cleaned_data)} entries with None 'latest_value'.")

    return cleaned_data

# 使用示例
json_file = "G://模型//data//news_data.json"
cleaned_data = clean_data(json_file)

# 你可以将清洗后的数据再保存到一个新的文件中，或者继续使用
with open("G://模型//data//cleaned_news_data.json", 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
