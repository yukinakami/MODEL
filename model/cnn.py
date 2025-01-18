import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=256):
        super(CNNEncoder, self).__init__()
        # 加载预训练的 ResNet18
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # 去掉最后的全连接层和 AdaptiveAvgPool2d
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 移除 AdaptiveAvgPool2d 和 fc
        
        # 获取最后一层卷积的输出通道数
        conv_out_channels = resnet.layer4[-1].conv1.out_channels  # 对于 ResNet18 是 512
        
        # 添加全局平均池化和自定义全连接层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (batch, channels, 1, 1)
        self.fc = nn.Linear(conv_out_channels, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 直接使用预处理后的输入图像
        features = self.resnet(x)  # 输出 (batch, 512, H, W)
        
        # 全局平均池化
        pooled_features = self.global_pool(features)  # 输出 (batch, 512, 1, 1)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)  # 展平为 (batch, 512)
        
        # 降维并激活
        output = self.relu(self.fc(flattened_features))
        return output

# 测试代码
if __name__ == "__main__":
    # 假设图片已被预处理为 (batch_size, 3, 224, 224) 的张量
    inputs = torch.randn(4, 3, 224, 224)  # 输入 4 张 224x224 的 RGB 图像
    encoder = CNNEncoder(pretrained=True)  # 定义编码器
    encoded_features = encoder(inputs)  # 编码特征
    print(f"Encoded Features Shape: {encoded_features.shape}")  # 输出特征维度
