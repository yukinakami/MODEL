import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_seq_len=10):
        super(CrossAttentionFusion, self).__init__()
        self.max_seq_len = max_seq_len  # 设定最大序列长度，作为扩充的标准
        
        # 设置用于文本和图像的线性层
        self.query_text = nn.Linear(input_dim, hidden_dim)
        self.key_image = nn.Linear(input_dim, hidden_dim)
        self.value_image = nn.Linear(input_dim, hidden_dim)

        self.query_image = nn.Linear(input_dim, hidden_dim)
        self.key_text = nn.Linear(input_dim, hidden_dim)
        self.value_text = nn.Linear(input_dim, hidden_dim)

        self.output_fc = nn.Linear(2 * hidden_dim, input_dim)

    def _expand_input(self, features, seq_len):
        """扩充输入，确保输入的形状为 (batch_size, seq_len, input_dim)"""
        batch_size, input_dim = features.shape

        # 扩展到指定的 seq_len，假设输入是 (batch_size, input_dim)
        expanded_features = features.unsqueeze(1).expand(batch_size, seq_len, input_dim)
        return expanded_features

    def forward(self, text_features, image_features):
        # 自动扩充输入特征
        seq_len_text = text_features.shape[1]  # 获取文本的序列长度
        seq_len_image = image_features.shape[1]  # 获取图像的序列长度

        # 通过扩展使得输入形状为 (batch_size, seq_len, input_dim)
        text_features = self._expand_input(text_features, self.max_seq_len)
        image_features = self._expand_input(image_features, self.max_seq_len)

        # 计算文本对图像的注意力
        query_text = self.query_text(text_features)  # (batch_size, seq_len_text, hidden_dim)
        key_image = self.key_image(image_features)  # (batch_size, seq_len_image, hidden_dim)
        value_image = self.value_image(image_features)  # (batch_size, seq_len_image, hidden_dim)

        attention_weights_text_to_image = F.softmax(
            torch.matmul(query_text, key_image.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_text, seq_len_image)
        attended_image = torch.matmul(attention_weights_text_to_image, value_image)  # (batch_size, seq_len_text, hidden_dim)

        # 计算图像对文本的注意力
        query_image = self.query_image(image_features)  # (batch_size, seq_len_image, hidden_dim)
        key_text = self.key_text(text_features)  # (batch_size, seq_len_text, hidden_dim)
        value_text = self.value_text(text_features)  # (batch_size, seq_len_text, hidden_dim)

        attention_weights_image_to_text = F.softmax(
            torch.matmul(query_image, key_text.transpose(-2, -1)), dim=-1
        )  # (batch_size, seq_len_image, seq_len_text)
        attended_text = torch.matmul(attention_weights_image_to_text, value_text)  # (batch_size, seq_len_image, hidden_dim)

        # 融合特征
        fused_features = torch.cat([attended_image.mean(dim=1), attended_text.mean(dim=1)], dim=-1)  # (batch_size, 2 * hidden_dim)
        #print(fused_features.shape)  # 打印调试信息，查看融合后的特征形状

        output_features = self.output_fc(fused_features)  # (batch_size, input_dim)
        return output_features


if __name__ == "__main__":
    # 假设我们有 1 个批次，每个批次中的文本和图像特征如下：
    input_text = torch.randn(1, 768)  # (batch_size=1, input_dim=768)
    input_image = torch.randn(4, 768)  # (batch_size=1, input_dim=768)

    # 定义模型，设置最大序列长度为 10
    encode = CrossAttentionFusion(input_dim=768, hidden_dim=256, max_seq_len=10)
    features = encode(input_text, input_image)
    #print(f'shape: {features.shape}')
