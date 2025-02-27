import torch
import torch.nn as nn
import os

class TimeSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, dropout):
        super(TimeSequenceModel, self).__init__()

        self.hidden_dim = hidden_dim # ���ز�ά��
        self.num_layers = num_layers # LSTM����
        self.seq_len = seq_len #���г���

        # LSTM��
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # ȫ���Ӳ�
        self.fc = nn.Linear(hidden_dim, 1) #���������һ�������������쳣���ĸ��ʣ�

    def forward(self, x):
        #print(f'x: {x.shape}')
        # ������״Ϊ (batch_size*seq_len, input_dim)
        # ������任Ϊ (batch_size, seq_len, input_dim)
        #batch_size = x.size(0) // self.seq_len #����batch_size
        #x = x.view(batch_size, self.seq_len, -1) 
        # ���� seq_len = 1
        seq_len = 1

        # �����ݱ���Ϊ (batch_size, seq_len, input_dim)
        x = x.view(x.size(0), seq_len, -1)  # x.size(0) Ϊ batch_size, -1 Ϊ�Զ��ƶ�����ά��
        #print(f'x2: {x.shape}')
        #�����κ�����ݴ��ݵ�LSTM�㣬�����״Ϊ (batch_size, seq_len, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x) #(h_n, c_n)��LSTM �����һ�������״̬ h_n ��ϸ��״̬ c_n

        #ѡ�����һ��ʱ�䲽�����
        out_put = self.fc(lstm_out[:, -1, :]) #ȡ���һ��ʱ�䲽����״̬Ϊ���

        return out_put