import torch
import torch.nn as nn
import torch.nn.functional as F

class DataCrossAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len=10):
        super(DataCrossAttentionFusion, self).__init__()
        self.max_seq_len = max_seq_len  # 设定最大序列长度，作为扩充的标准
        
        # 设置用于文本和图像的线性层
        self.query_fused = nn.Linear(input_dim, hidden_dim)
        self.key_data = nn.Linear(input_dim, hidden_dim)
        self.value_data = nn.Linear(input_dim, hidden_dim)

        self.query_data = nn.Linear(input_dim, hidden_dim)
        self.key_fused = nn.Linear(input_dim, hidden_dim)
        self.value_fused = nn.Linear(input_dim, hidden_dim)

        self.output_fc = nn.Linear(2 * hidden_dim, input_dim)

    def _expand_input(self, features, seq_len):
        """扩充输入，确保输入的形状为 (batch_size, seq_len, input_dim)"""
        batch_size, input_dim = features.shape

        # 扩展到指定的 seq_len，假设输入是 (batch_size, input_dim)
        expanded_features = features.unsqueeze(1).expand(batch_size, seq_len, input_dim)
        return expanded_features

    def forward(self, fused_features, data_features):
        # 自动扩充输入特征
        seq_len_fused = fused_features.shape[1]  # 获取文本的序列长度
        seq_len_data = data_features.shape[1]  # 获取图像的序列长度

        # 通过扩展使得输入形状为 (batch_size, seq_len, input_dim)
        fused_features = self._expand_input(fused_features, self.max_seq_len)
        data_features = self._expand_input(data_features, self.max_seq_len)

        # 计算文本对图像的注意力
        query_fused = self.query_fused(fused_features)  # (batch_size, seq_len_text, hidden_dim)
        key_data = self.key_data(data_features)  # (batch_size, seq_len_image, hidden_dim)
        value_data = self.value_data(data_features)  # (batch_size, seq_len_image, hidden_dim)

        attention_weights_fused_to_data = F.softmax(
            torch.matmul(query_fused, key_data.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_text, seq_len_image)
        attended_data = torch.matmul(attention_weights_fused_to_data, value_data)  # (batch_size, seq_len_text, hidden_dim)

        # 计算图像对文本的注意力
        query_data = self.query_data(data_features)  # (batch_size, seq_len_image, hidden_dim)
        key_fused = self.key_fused(fused_features)  # (batch_size, seq_len_text, hidden_dim)
        value_fused = self.value_fused(fused_features)  # (batch_size, seq_len_text, hidden_dim)

        attention_weights_data_to_fused = F.softmax(
            torch.matmul(query_data, key_fused.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_image, seq_len_text)
        attended_fused = torch.matmul(attention_weights_data_to_fused, value_fused)  # (batch_size, seq_len_image, hidden_dim)

        # 融合特征
        final_fused_features = torch.cat([attended_data.mean(dim=1), attended_fused.mean(dim=1)], dim=-1)  # (batch_size, 2 * hidden_dim)
        print(final_fused_features.shape)  # 打印调试信息，查看融合后的特征形状

        output_features = self.output_fc(final_fused_features)  # (batch_size, input_dim)
        return output_features


if __name__ == "__main__":
    # 假设我们有 1 个批次，每个批次中的文本和图像特征如下：
    input_text = torch.randn(1, 768)  # (batch_size=1, input_dim=768)
    input_image = torch.randn(4, 768)  # (batch_size=1, input_dim=768)

    # 定义模型，设置最大序列长度为 10
    encode = DataCrossAttentionFusion(input_dim=768, hidden_dim=256, max_seq_len=10)
    features = encode(input_text, input_image)
    print(f'shape: {features.shape}')
