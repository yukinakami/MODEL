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
        # �ı�������
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # ͼ�������
        self.image_encoder = CNNEncoder(input_dim=image_input_dim)  # 128 ���ںϲ���м�ά��

    def forward(self, text, image):
        # ��ȡ�ı�����
        text_features = self.text_encoder.encode(text)  # ��� (batch_size, 768)
        # ��ȡͼ������
        image_features = self.image_encoder(image)  # ��� (batch_size, 128)
        
        #������Ҫ��ͼ��ά����Ϊͼ���ά��̫����
        
        # �ں�����
        fusion_layer = CrossAttentionFusion(input_dim=512, hidden_dim=256)
        fused_features = fusion_layer(text_features, image_features)
