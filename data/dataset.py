import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from transformers import BertTokenizer
from torchvision import transforms

# 允许加载损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultimodalDataset(Dataset):
    def __init__(self, json_file, max_length=512):
        with open(json_file, 'r', encoding='utf-') as f:
            self.data = json.load(f)
        
        # 初始化 BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 定义图像预处理操作
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 统一图像尺寸
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取数据
        entry = self.data[idx]
        text = entry['text']
        image_path = entry['image_path']
        data = entry['latest_value']
        
        # 处理文本：转换为张量形式
        text_inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True,
            max_length=self.max_length, 
            return_tensors='pt')
        
        # 处理图像：加载图像并应用预处理
        try:
            image = Image.open(image_path).convert('RGB')  # 打开图片并转换为 RGB 模式
            image_tensor = self.transform(image)  # 应用预处理操作
        except OSError as e:
            # 如果遇到损坏的图像，打印路径并加载默认图像
            print(f"Skipping corrupted image: {image_path}")
            # 直接使用写死的默认图像路径
            default_image_path = 'G://模型//data//1.jpg'
            image = Image.open(default_image_path).convert('RGB')
            image_tensor = self.transform(image)  # 应用预处理操作
        
        # 将时序数据转换为张量
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # 返回处理后的文本、图像和时序数据
        text = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask']
        }
        image = image_tensor
        data = data_tensor
        
        return text, image, data
