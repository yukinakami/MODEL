import torch
import torch.nn as nn
import torch.nn.functional as F

from model.data_encoder import DataEncoder

class AudioDataCrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AudioDataCrossAttention, self).__init__()


        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj= nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, audio_features, time_features):
        #����ע��������
        query = self.query_proj(audio_features).unsqueeze(1) #.unsqueeze(dim) ����ָ����ά�� dim �ϲ���һ����СΪ 1 ��ά��
        key = self.key_proj(time_features).unsqueeze(1)
        value = self.value_proj(time_features).unsqueeze(1)

        # ע��������
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (key.size(-1) ** 0.5)
        #ע������Ȩ
        attention_weights = self.softmax(attention_scores)

        #�����ں�
        attended_features = torch.bmm(attention_weights, value) #��������˷�
        fused_features = attended_features.squeeze(1) + audio_features
        return fused_features
        