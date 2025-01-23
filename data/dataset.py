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
        with open(json_file, 'r', encoding='utf-8') as f:
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

        # 计算全局的均值和标准差（如果是时序数据的目标）
        self.mean = None
        self.std = None
        if self.data:
            all_values = [entry['latest_value'] for entry in self.data]
            all_values = np.array(all_values)
            self.mean = np.mean(all_values)
            self.std = np.std(all_values)

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
        image = Image.open(image_path).convert('RGB')  # 打开图片并转换为 RGB 模式
        image_tensor = self.transform(image)  # 应用预处理操作

        # 将时序数据转换为张量
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 标准化目标标签
        if self.mean is not None and self.std is not None:
            normalized_target = (data_tensor - self.mean) / self.std
        else:
            normalized_target = data_tensor  # 如果没有全局标准化参数，则不标准化

        # 将时序数据和目标标签的形状保持一致
        data_tensor = data_tensor.view(1, 1)
        normalized_target = normalized_target.view(1, 1)
        
        # 返回处理后的文本、图像和时序数据
        text = {
            'input_ids': text_inputs['input_ids'].squeeze(0),  # 通过 squeeze 去除批次维度
            'attention_mask': text_inputs['attention_mask'].squeeze(0)
        }
        image = image_tensor
        data = data_tensor
        target = normalized_target
        
        return text, image, data, target
