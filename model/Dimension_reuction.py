#使用LSTM模型对图像编码进行降维

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LSTM(nn.Module):
    def __init__(self, input_channel, bert_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_channel, #输入通道数（cnn的输出维度）
            hidden_size = bert_dim,  #降维后的维度（bert的输出维度）
            num_layers = 1, #lstm的层数 = 1
            batch_first = False, #时间步在第0维
            bidirectional = False #单向lstm
            )
        
    def forward(self, cnn_output):
        # cnn_output: (batch_size, seq_len, h, w)
        batch_size, seq_len, h, w = cnn_output.shape
        # 将cnn的输出展平
        cnn_output = cnn_output.view(batch_size, seq_len, -1)  #view是调整张量形状的函数，-1的意思是自动计算这个维度的大小
        #调整为符合lstm输入的顺序
        cnn_output = cnn_output.permute(2, 0, 1) #permute是调整张量维度顺序的函数

        #使用lstm降维
        lstm_output,_ = self.lstm(cnn_output) #,_表示只取第一个返回值,即lstm_output,忽略lstm的隐藏状态hidden_state和细胞状态cell_state
        #取最后一个时间步的输出
        final_output = lstm_output[-1, :, :] #输出为(batch_size, bert_dim)
        return final_output
    
if __name__ == "__main__":
    # 假设图片已被预处理为 (batch_size, 3, 224, 224) 的张量
    inputs = torch.randn(4, 512, 7, 7)  # 输入 4 张 224x224 的 RGB 图像
    encoder = LSTM(input_channel=512, bert_dim=768)  # 定义编码器
    encoded_features = encoder(inputs)  # 编码特征
    print(f"Encoded Features Shape: {encoded_features.shape}")  # 输出特征维度
