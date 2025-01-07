import torch
import torch.nn as nn
import os

from model.cnn import CNNEncoder
from model.bert import BertEncoder
from model.crossattention import CrossAttentionFusion
from model.Dimension_reuction import LSTM
from model.data_encoder import DataEncoder
from model.data_crossattention import DataCrossAttention
from model.audio_feacture import AudioEncoder
from model.crossattention_image_data import ImageDataCrossAttention
from model.crossattention_text_data import TextDataCrossAttention
from model.crossattention_audio_data import AudioDataCrossAttention

import numpy as np

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_input_dim=768, time_input_dim=768, audio_input_dim=768):
        super(MultimodalModel, self).__init__()
        # �ı�������
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # ͼ�������
        self.image_encoder = CNNEncoder(input_dim=image_input_dim)
        # ʱ�������
        self.time_encoder = DataEncoder(input_dim=time_input_dim)
        # ��Ƶ������
        self.audio_encoder = AudioEncoder(input_dim=audio_input_dim)

    def forward(self, text, image, data, audio):
        # ��ȡ�ı�����
        text_features = self.text_encoder.encode(text)  # ��� (batch_size, 768)
        # ��ȡͼ������
        image_features = self.image_encoder(image)  
        #��ȡʱ������
        time_features =self.time_encoder(data)
        #��ȡ��Ƶ����
        audio_features = self.audio_encoder(audio)
        
        if image_features is not None:
            #������Ҫ��ͼ��ά����Ϊͼ���ά��̫����
            reducer = LSTM(input_channel=512, bert_dim=768)
            image_feacture_reduction = reducer(image_features)
        else:
            image_feacture_reduction = None
        
        #��ͬģ̬��ʱ����
        if text_features is not None and image_features is not None: #ͼ��+ʱ��
            # �ں�ͼ������
            fusion_layer = CrossAttentionFusion(input_dim=768)
            fused_features = fusion_layer(text_features, image_feacture_reduction)
            # �ں�ʱ������
            data_fusion_layer = DataCrossAttention(input_dim=768)
            data_fusion_features = data_fusion_layer(fused_features, time_features)
            return data_fusion_features
        
        elif image_feacture_reduction is not None and text_features is None: #ͼ��+ʱ��
            data_fusion_layer = ImageDataCrossAttention(input_dim=768)
            data_fusion_features = data_fusion_layer(image_feacture_reduction, time_features)
            return data_fusion_features
        
        elif text_features is not None and image_feacture_reduction is None: #�ı�+ʱ��
            data_fusion_layer = TextDataCrossAttention(input_dim=768)
            data_fusion_features = data_fusion_layer(text_features, time_features)
            return data_fusion_features
        
        elif audio_features is not None: #��Ƶ+ʱ��
            data_fusion_layer = AudioDataCrossAttention(input_dim=768)
            data_fusion_features = data_fusion_layer(audio_features, time_features)
            return data_fusion_features
        
        else:
            return time_features