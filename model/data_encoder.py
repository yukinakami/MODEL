import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DataEncoder(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=1):
        super(DataEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size = 1, #输入通道数（cnn的输出维度）
            hidden_size = hidden_dim,  #编码后的维度（bert的输出维度）
            num_layers = num_layers, #lstm的层数 = 1
            batch_first = False, #时间步在第0维
            )
        
    def forward(self, x):
        #print(f'x shape: {x.shape}')
        #print(f'x: {x}')
        # 如果 batch_first=False, 那么 x 的形状需要是 (seq_len, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # 转置为 (seq_len, batch_size, input_dim)
        print(f'Input x after permute: {x.shape}')  # 查看转置后的数据形状
        lstm_out, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]

if __name__ == "__main__":
    x = torch.randn(1,32,1)
    encode = DataEncoder(hidden_dim=768)
    a = encode(x)
    print(a.shape)
