import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个基本的交叉注意力层
class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim_text, input_dim_image, input_dim_time, attention_dim, seq_len):
        super(CrossAttentionModel, self).__init__()

        self.seq_len = seq_len  # 存储序列长度
        
        # 定义查询、键、值的线性变换
        self.query_text = nn.Linear(input_dim_text, attention_dim)
        self.query_image = nn.Linear(input_dim_image, attention_dim)
        self.query_time = nn.Linear(input_dim_time, attention_dim)

        self.key_text = nn.Linear(input_dim_text, attention_dim)
        self.key_image = nn.Linear(input_dim_image, attention_dim)
        self.key_time = nn.Linear(input_dim_time, attention_dim)

        self.value_text = nn.Linear(input_dim_text, attention_dim)
        self.value_image = nn.Linear(input_dim_image, attention_dim)
        self.value_time = nn.Linear(input_dim_time, attention_dim)

        self.attn_layer = nn.MultiheadAttention(attention_dim, num_heads=1, batch_first=True)

    def _expand_input(self, features):
        """扩展输入，确保输入的形状为 (batch_size, seq_len, input_dim)"""
        batch_size, input_dim = features.shape

        # 扩展到指定的 seq_len，假设输入是 (batch_size, input_dim)
        expanded_features = features.unsqueeze(1).expand(batch_size, self.seq_len, input_dim)
        return expanded_features
        
    def forward(self, text_features, image_features, time_features):
    # 对原始输入特征进行扩展
        text_features = self._expand_input(text_features)
        image_features = self._expand_input(image_features)
        time_features = self._expand_input(time_features)

        # 为文本、图像和时序数据生成查询、键和值
        query_text = self.query_text(text_features)
        query_image = self.query_image(image_features)
        query_time = self.query_time(time_features)

        key_text = self.key_text(text_features)
        key_image = self.key_image(image_features)
        key_time = self.key_time(time_features)

        value_text = self.value_text(text_features)
        value_image = self.value_image(image_features)
        value_time = self.value_time(time_features)

        # 将查询、键和值合并（将它们按需要的方式结合）
        queries = torch.cat([query_text, query_image, query_time], dim=1)
        keys = torch.cat([key_text, key_image, key_time], dim=1)
        values = torch.cat([value_text, value_image, value_time], dim=1)

        # 使用线性层调整维度
        queries = self.fc(queries)
        keys = self.fc(keys)
        values = self.fc(values)

        # 进行交叉注意力
        attn_output, _ = self.attn_layer(queries, keys, values)

        # 返回注意力结果
        return attn_output

