import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1, output_dim=1):
        super(TransformerTimeSeries, self).__init__()
        
        # 输入嵌入层（如果输入已经是768维，可以省略）
        self.embedding = nn.Linear(input_dim, input_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层（用于时序预测）
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        seq_len = 1
        x = x.view(x.size(0), seq_len, -1)  # x.size(0) 为 batch_size, -1 为自动推断输入维度
        # 输入形状: (batch_size, sequence_length, input_dim)
        
        # 嵌入层（如果输入已经是768维，可以省略）
        #x = self.embedding(x)
        
        # Transformer需要输入形状为 (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 恢复形状为 (batch_size, sequence_length, input_dim)
        x = x.permute(1, 0, 2)
        
        # 输出层（预测每个时间步的值）
        output = self.output_layer(x)
        
        # 输出形状: (batch_size, sequence_length, output_dim)
        return output

# 示例用法
if __name__ == "__main__":
    # 假设输入数据形状为 (batch_size=32, sequence_length=10, input_dim=768)
    batch_size = 32
    sequence_length = 10
    input_dim = 768
    x = torch.randn(batch_size, sequence_length, input_dim)
    
    # 初始化模型
    model = TransformerTimeSeries(input_dim=input_dim, output_dim=1)
    
    # 前向传播
    output = model(x)
    print("Output shape:", output.shape)  # 应该输出: (32, 10, 1)a