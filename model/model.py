import torch
import torch.nn as nn
import os

from model.cnn import CNNEncoder
from model.bert import BertEncoder
from model.crossattention import CrossAttentionFusion

import numpy as np

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_input_dim=768):
        super(MultimodalModel, self).__init__()
        # 文本编码器
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # 图像编码器
        self.image_encoder = CNNEncoder(input_dim=image_input_dim)  # 128 是融合层的中间维度

    def forward(self, text, image):
        # 获取文本特征
        text_features = self.text_encoder.encode(text)  # 输出 (batch_size, 768)
        # 获取图像特征
        image_features = self.image_encoder(image)  # 输出 (batch_size, 128)
        
        #这里需要对图像降维，因为图像的维度太高了
        
        # 融合特征
        fusion_layer = CrossAttentionFusion(input_dim=512, hidden_dim=256)
        fused_features = fusion_layer(text_features, image_features)
