import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DataEncoder:
    def __init__(self, input_dim, hidden_dim=768, num_layers=1):
        super(DataEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, #����ͨ������cnn�����ά�ȣ�
            hidden_size = hidden_dim,  #��ά���ά�ȣ�bert�����ά�ȣ�
            num_layers = num_layers, #lstm�Ĳ��� = 1
            batch_first = False, #ʱ�䲽�ڵ�0ά
            )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (hidden_state,_) = self.lstm(x)
        #�������һ��ʱ�䲽������״̬
        return hidden_state[-1]
        #���ά��Ϊ(batch_size, hidden_dim)