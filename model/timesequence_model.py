import torch
import torch.nn as nn
import os

class TimeSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, dropout):
        super(TimeSequenceModel, self).__init__()

        self.hidden_dim = hidden_dim # 隐藏层维度
        self.num_layers = num_layers # LSTM层数
        self.seq_len = seq_len #序列长度

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, 1) #假设输出是一个标量（例如异常检测的概率）

    def forward(self, x):
        #print(f'x: {x.shape}')
        # 输入形状为 (batch_size*seq_len, input_dim)
        # 将输入变换为 (batch_size, seq_len, input_dim)
        #batch_size = x.size(0) // self.seq_len #计算batch_size
        #x = x.view(batch_size, self.seq_len, -1) 
        # 假设 seq_len = 1
        seq_len = 1

        # 将数据变形为 (batch_size, seq_len, input_dim)
        x = x.view(x.size(0), seq_len, -1)  # x.size(0) 为 batch_size, -1 为自动推断输入维度
        #print(f'x2: {x.shape}')
        #将变形后的数据传递到LSTM层，输出形状为 (batch_size, seq_len, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x) #(h_n, c_n)：LSTM 在最后一层的隐藏状态 h_n 和细胞状态 c_n

        #选择最后一个时间步的输出
        out_put = self.fc(lstm_out[:, -1, :]) #取最后一个时间步的隐状态为输出

        return out_put