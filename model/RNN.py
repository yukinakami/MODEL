import torch
import torch.nn as nn
import os

class TimeSequenceModell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, dropout):
        super(TimeSequenceModell, self).__init__()

        self.hidden_dim = hidden_dim  # ���ز�ά��
        self.num_layers = num_layers  # GRU����
        self.seq_len = seq_len  # ���г���

        # GRU��
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # ȫ���Ӳ�
        self.fc = nn.Linear(hidden_dim, 1)  # ���������һ�������������쳣���ĸ��ʣ�

    def forward(self, x):
        # ���� seq_len = 1
        seq_len = 1
        # �����ݱ���Ϊ (batch_size, seq_len, input_dim)
        x = x.view(x.size(0), seq_len, -1)  # x.size(0) Ϊ batch_size, -1 Ϊ�Զ��ƶ�����ά��
        # �����κ�����ݴ��ݵ�GRU�㣬�����״Ϊ (batch_size, seq_len, hidden_dim)
        rnn_out, hn = self.rnn(x)  # (hn): GRU �����һ�������״̬
        # ѡ�����һ��ʱ�䲽�����
        out_put = self.fc(rnn_out[:, -1, :])  # ȡ���һ��ʱ�䲽����״̬Ϊ���

        return out_put
