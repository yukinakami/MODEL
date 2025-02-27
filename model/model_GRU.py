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
        # �ı�������
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # ͼ�������
        self.image_encoder = CNNEncoder(pretrained=image_pretrained)
        # ���ݱ�����
        self.data_encoder = DataEncoder(hidden_dim=768)
        # ͼ��ά
        self.reducer = LSTM(input_channel=512, bert_dim=768)
        # ͼ�Ľ���
        #self.crossattention = CrossAttentionFusion(input_dim=768, hidden_dim=256)
        #ʱ����
        #self.data_crossattention = DataCrossAttentionFusion(input_dim=768, hidden_dim=256)
        #ģ̬�ں�
        self.crossattention = ModalFusionModel(text_dim=768, image_dim=768, data_dim=768, hidden_dim=768, attention_heads=8, num_attention_layers=3)

        # ʱ��ģ��
        self.timesequence = TimeSequenceModell(input_dim=768, hidden_dim=hidden_dim, num_layers=4, seq_len=seq_len, dropout=dropout)
        
    def forward(self, text, image, data):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        # ��ȡ�ı�����
        text_features = self.text_encoder.encode(input_ids, attention_mask)  # ��� (batch_size, 768)
        # ��ȡͼ������
        image_features = self.image_encoder(image)       
        #������Ҫ��ͼ��ά����Ϊͼ���ά��̫����
        image_feacture_reduction = self.reducer(image_features)
        # ��������
        data_features = self.data_encoder(data)
        # �ں�����
        fused_features = self.crossattention(text_features, image_feacture_reduction, data_features)
        # ʱ��ģ��
        output = self.timesequence(fused_features)

        return output
