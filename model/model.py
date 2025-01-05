import torch
import torch.nn as nn
import os

from model.cnn import CNNEncoder
from model.bert import BertEncoder
from model.crossattention import CrossAttentionFusion
from model.Dimension_reuction import LSTM
from model.data_encoder import DataEncoder
from model.data_crossattention import DataCrossAttention

import numpy as np

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_input_dim=768, time_input_dim=768):
        super(MultimodalModel, self).__init__()
        # 文本编码器
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # 图像编码器
        self.image_encoder = CNNEncoder(input_dim=image_input_dim)
        # 时序编码器
        self.time_encoder = DataEncoder(input_dim=time_input_dim)

    def forward(self, text, image, data):
        # 获取文本特征
        text_features = self.text_encoder.encode(text)  # 输出 (batch_size, 768)
        # 获取图像特征
        image_features = self.image_encoder(image)  
        #获取时序特征
        time_features =self.time_encoder(data)
        
        #这里需要对图像降维，因为图像的维度太高了
        reducer = LSTM(input_channel=512, bert_dim=768)
        image_feacture_reduction = reducer(image_features)
        
        # 融合图文特征
        fusion_layer = CrossAttentionFusion(input_dim=768)
        fused_features = fusion_layer(text_features, image_feacture_reduction)
        # 融合时序特征
        data_fusion_layer = DataCrossAttention(input_dim=768)
        data_fusion_features = data_fusion_layer(fused_features, time_features)
