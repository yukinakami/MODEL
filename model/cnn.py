import torch
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNEncoder, self).__init__()
        # 加载预训练的 ResNet18
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # 去掉最后的全连接层和 AdaptiveAvgPool2d
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 移除 AdaptiveAvgPool2d 和 fc
        
    def forward(self, x):
        # 直接使用预处理后的输入图像
        features = self.resnet(x)  # 输出 (batch, 512, H, W)
        
        return features

# 测试代码
if __name__ == "__main__":
    # 假设图片已被预处理为 (batch_size, 3, 224, 224) 的张量
    inputs = torch.randn(4, 3, 224, 224)  # 输入 4 张 224x224 的 RGB 图像
    encoder = CNNEncoder(pretrained=True)  # 定义编码器
    encoded_features = encoder(inputs)  # 编码特征
    print(f"Encoded Features Shape: {encoded_features.shape}")  # 输出特征维度
