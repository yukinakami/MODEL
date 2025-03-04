import os
import json
import pandas as pd

data_root = 'G://news'
data_list = []

# 读取iron.xlsx文件并将日期和最新值列提取出来
def get_iron_data(excel_path):
    df = pd.read_excel(excel_path)
    # 假设 '日期' 列为日期列，'最新值' 列为你想要添加的列
    iron_data = {}
    for index, row in df.iterrows():
        iron_data[row['日期']] = row['最新值']
    return iron_data

# 保存为JSON文件
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 读取iron.xlsx中的数据
iron_data = get_iron_data('G://iron.xlsx')

# 遍历数据
for date_folder in os.listdir(data_root):
    date_path = os.path.join(data_root, date_folder)
    if os.path.isdir(date_path):
        for news_folder in os.listdir(date_path):
            news_path = os.path.join(date_path, news_folder)
            if os.path.isdir(news_path):
                # 读取文本数据
                text_files = [f for f in os.listdir(news_path) if f.endswith('.txt')]
                text_contents = []
                for text_file in text_files:
                    text_path = os.path.join(news_path, text_file)
                    with open(text_path, 'r', encoding='utf-8') as file:
                        text_contents.append(file.read())
                
                # 读取图片数据
                image_files = [f for f in os.listdir(news_path) if f.endswith(('.jpg', '.jpeg', '.png', '.mp4'))]
                image_paths = [os.path.join(news_path, img) for img in image_files]
                
                # 获取对应日期的最新值
                latest_value = iron_data.get(date_folder, None)
                
                # 如果有文本内容
                if text_contents:
                    # 遍历所有文本内容
                    for text_content in text_contents:
                        # 如果有图片，生成每个文本和每个图片的组合数据
                        if image_paths:
                            for image_path in image_paths:
                                data_list.append({
                                    "image_path": image_path,  # 配对当前图片
                                    "text": text_content,      # 配对当前文本
                                    "news_title": news_folder,
                                    "news_date": date_folder,
                                    "latest_value": latest_value  # 添加最新值字段
                                })
                        # 如果没有图片，则生成包含文本但没有图片的条目
                        else:
                            data_list.append({
                                "image_path": "G://news//1.jpg",  # 默认图片（无图片的情况）
                                "text": text_content,
                                "news_title": news_folder,
                                "news_date": date_folder,
                                "latest_value": latest_value  # 添加最新值字段
                            })

# 在数据处理结束后，保存一次JSON文件
save_to_json(data_list, 'G://模型//data//news_data.json')

print("处理完成，JSON文件已保存")
