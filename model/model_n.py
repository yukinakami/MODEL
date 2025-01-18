import torch
import torch.nn as nn
import os

from model.cnn import CNNEncoder
from model.bert import BertEncoder
from model.crossattention import CrossAttentionFusion
from model.Dimension_reuction import LSTM
from model.data_encoder import DataEncoder
from model.data_crossattention import DataCrossAttention
from model.timesequence_model import TimeSequenceModel
from model.data_crossattention import DataCrossAttention

import numpy as np

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_pretrained=True, time_input_dim=768, seq_len=10, hidden_dim=128, num_layers=2, dropout=0.1):
        super(MultimodalModel, self).__init__()
        # 文本编码器
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # 图像编码器
        self.image_encoder = CNNEncoder(pretrained=image_pretrained)
        # 数据编码器
        self.data_encoder = DataEncoder(input_dim=768)
        # 图像降维
        self.reducer = LSTM(input_channel=512, bert_dim=768)
        # 图文交融
        self.crossattention = CrossAttentionFusion(input_dim=768, hidden_dim=256)
        #时序交融
        self.data_crossattention = DataCrossAttention(input_dim=768, hidden_dim=256)
        # 时序模型
        self.timesequence = TimeSequenceModel(input_dim=768, hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len, dropout=dropout)
        

    def forward(self, text, image, data):
        # 获取文本特征
        text_features = self.text_encoder.encode(text)  # 输出 (batch_size, 768)
        # 获取图像特征
        image_features = self.image_encoder(image)       
        #这里需要对图像降维，因为图像的维度太高了
        image_feacture_reduction = self.reducer(image_features)
        # 数据特征
        data_features = self.data_encoder(data)
        # 融合图文特征
        fused_features = self.crossattention(text_features, image_feacture_reduction)
        # 时序融合
        data_image_text_features = self.data_crossattention(fused_features, data_features)
        # 时序模型
        output = self.timesequence(data_image_text_features)

        return output
