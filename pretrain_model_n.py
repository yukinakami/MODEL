import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MultimodalDataset  # 假设 Dataset 存储在 dataset.py 文件中
from model.model_n import MultimodalModel  # 假设模型存储在 model.py 文件中
from sklearn.model_selection import train_test_split
import json

# 数据集
dataset = MultimodalDataset("G://模型//data//cleaned_news_data2.json")

# 切分数据集为训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 初始化模型
model = MultimodalModel(text_model_name='bert-base-uncased', image_pretrained=True, hidden_dim=256)

# 选择损失函数和优化器
criterion = nn.MSELoss()  # 如果是回归任务，使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 设备设置（如果有GPU可用，使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    for text, image, data in train_loader:
        # 将数据移动到 GPU/CPU
        text = text.to(device)
        image = image.to(device)
        data = data.to(device)

        # 前向传播
        output = model(text, image, data)

        # 假设目标标签是时序数据（例如价格）
        target = data.to(device)

        # 计算损失
        loss = criterion(output, target)
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # 验证过程（每个 epoch 结束时）
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():  # 在验证时不需要计算梯度
        for text, image, data in val_loader:
            text = text.to(device)
            image = image.to(device)
            data = data.to(device)

            # 前向传播
            output = model(text, image, data)

            # 假设目标标签是时序数据（例如价格）
            target = data.to(device)

            # 计算损失
            loss = criterion(output, target)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")

# 保存模型
torch.save(model.state_dict(), "multimodal_model.pth")
