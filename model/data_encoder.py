import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DataEncoder:
    def __init__(self, input_dim, hidden_dim=768, num_layers=1):
        super(DataEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, #输入通道数（cnn的输出维度）
            hidden_size = hidden_dim,  #降维后的维度（bert的输出维度）
            num_layers = num_layers, #lstm的层数 = 1
            batch_first = False, #时间步在第0维
            )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, (hidden_state,_) = self.lstm(x)
        #返回最后一个时间步的隐藏状态
        return hidden_state[-1]
        #输出维度为(batch_size, hidden_dim)