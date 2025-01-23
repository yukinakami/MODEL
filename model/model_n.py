import torch
import torch.nn as nn
import torch.nn.init as init
from model.cnn import CNNEncoder
from model.bert import BertEncoder
from model.crossattention import CrossAttentionFusion
from model.Dimension_reuction import LSTM
from model.data_encoder import DataEncoder
from model.data_crossattention import DataCrossAttentionFusion
from model.timesequence_model import TimeSequenceModel

class MultimodalModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_pretrained=True, seq_len=10, hidden_dim=128, num_layers=2, dropout=0.1):
        super(MultimodalModel, self).__init__()
        # 文本编码器
        self.text_encoder = BertEncoder(model_name=text_model_name)
        # 图像编码器
        self.image_encoder = CNNEncoder(pretrained=image_pretrained)
        # 数据编码器
        self.data_encoder = DataEncoder(hidden_dim=768)
        # 图像降维
        self.reducer = LSTM(input_channel=512, bert_dim=768)
        # 图文交融
        self.crossattention = CrossAttentionFusion(input_dim=768, hidden_dim=256)
        # 时序交融
        self.data_crossattention = DataCrossAttentionFusion(input_dim=768, hidden_dim=256)
        # 时序模型
        self.timesequence = TimeSequenceModel(input_dim=768, hidden_dim=hidden_dim, num_layers=num_layers, seq_len=seq_len, dropout=dropout)
        
        # 调用初始化函数
        self._initialize_weights()

    def _initialize_weights(self):

        # 初始化图像编码器CNN的卷积层权重
        for module in self.image_encoder.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')  # 使用He初始化
                if module.bias is not None:
                    init.zeros_(module.bias)  # 将偏置初始化为0

        # 初始化LSTM降维层的权重
        for name, param in self.reducer.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param, nonlinearity='relu')  # 对LSTM层的权重使用He初始化
            elif 'bias' in name:
                init.zeros_(param)  # 将偏置初始化为0

        # 对CrossAttentionFusion和DataCrossAttentionFusion的权重进行初始化
        for module in [self.crossattention, self.data_crossattention]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)  # 使用Xavier初始化
                elif 'bias' in name:
                    init.zeros_(param)  # 偏置初始化为0

        # 初始化时序模型的LSTM层
        for name, param in self.timesequence.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param, nonlinearity='relu')  # 对LSTM层的权重使用He初始化
            elif 'bias' in name:
                init.zeros_(param)  # 偏置初始化为0

    def forward(self, text, image, data):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        # 获取文本特征
        text_features = self.text_encoder.encode(input_ids, attention_mask)  # 输出 (batch_size, 768)
        # 获取图像特征
        image_features = self.image_encoder(image)       
        # 这里需要对图像降维，因为图像的维度太高了
        image_feacture_reduction = self.reducer(image_features)
        # 数据特征
        data_features = self.data_encoder(data)
        # 融合图文特征
        fused_features = self.crossattention(text_features, image_feacture_reduction)
        # 时序融合
        #data_image_text_features = self.data_crossattention(fused_features, data_features)
        # 时序模型
        output = self.timesequence(fused_features)

        return output

