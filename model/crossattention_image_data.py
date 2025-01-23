import torch
import torch.nn as nn
import torch.nn.functional as F

# 简单的交叉注意力层定义
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_heads=8):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=attention_heads)
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, query, key, value):
        # query, key, value 是三种不同模态的特征向量
        attn_output, _ = self.attn(query, key, value)
        return self.layer_norm(attn_output + query)  # 残差连接

# 融合模型
class ModalFusionModel(nn.Module):
    def __init__(self, text_dim, image_dim, data_dim, hidden_dim, attention_heads=8, num_attention_layers=2):
        super(ModalFusionModel, self).__init__()
        
        # 假设每种模态特征已经是一个向量，映射到统一的维度
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        self.data_fc = nn.Linear(data_dim, hidden_dim)
        
        # 交叉注意力层（多层交叉注意力）
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, attention_heads) for _ in range(num_attention_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)  # 假设最终输出是一个标量（如回归问题）
    
    def forward(self, text, image, data):
        # 将输入模态特征转化为隐藏维度
        text_feat = self.text_fc(text)
        image_feat = self.image_fc(image)
        data_feat = self.data_fc(data)
        
        # 将它们堆叠为batch中的多个模态特征
        combined_feats = torch.stack([text_feat, image_feat, data_feat], dim=0)  # (3, batch_size, hidden_dim)
        
        # 多层交叉注意力机制融合模态
        for cross_attention in self.cross_attention_layers:
            fusion_output = cross_attention(combined_feats, combined_feats, combined_feats)
        
        # 融合后的特征取平均池化，作为最终的特征表示
        fusion_output = fusion_output.mean(dim=0)  # (batch_size, hidden_dim)
        
        # 最终输出
        output = self.fc_out(fusion_output)
        return output
