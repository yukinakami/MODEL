import torch
import torch.nn as nn
import os

from model.cnn import CNNEncoder
from model.bert import BertEncoder
#from model.crossattention import CrossAttentionFusion
from model.Dimension_reuction import LSTM
from model.data_encoder import DataEncoder
#from model.data_crossattention import DataCrossAttentionFusion
from model.GRU import TimeSequenceModell
from model.crossattention_image_data import ModalFusionModel

import numpy as np

class MultimodalModel_GRU(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_pretrained=True, seq_len=10, hidden_dim=128, num_layers=2, dropout=0.1):
        super(MultimodalModel_GRU, self).__init__()
        # 文本编码器
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # 图像编码器
        self.image_encoder = CNNEncoder(pretrained=image_pretrained)
        # 数据编码器
        self.data_encoder = DataEncoder(hidden_dim=768)
        # 图像降维
        self.reducer = LSTM(input_channel=512, bert_dim=768)
        # 图文交融
        #self.crossattention = CrossAttentionFusion(input_dim=768, hidden_dim=256)
        #时序交融
        #self.data_crossattention = DataCrossAttentionFusion(input_dim=768, hidden_dim=256)
        #模态融合
        self.crossattention = ModalFusionModel(text_dim=768, image_dim=768, data_dim=768, hidden_dim=768, attention_heads=8, num_attention_layers=3)

        # 时序模型
        self.timesequence = TimeSequenceModell(input_dim=768, hidden_dim=hidden_dim, num_layers=4, seq_len=seq_len, dropout=dropout)
        
    def forward(self, text, image, data):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        # 获取文本特征
        text_features = self.text_encoder.encode(input_ids, attention_mask)  # 输出 (batch_size, 768)
        # 获取图像特征
        image_features = self.image_encoder(image)       
        #这里需要对图像降维，因为图像的维度太高了
        image_feacture_reduction = self.reducer(image_features)
        # 数据特征
        data_features = self.data_encoder(data)
        # 融合特征
        fused_features = self.crossattention(text_features, image_feacture_reduction, data_features)
        # 时序模型
        output = self.timesequence(fused_features)

        return output
