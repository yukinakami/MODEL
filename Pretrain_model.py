import torch
import torch.nn as nn
import os
import torch.optim as optim
import json
from torch.utils.data import DataLoader, Dataset
from model.model import MultimodalModel

# 定义数据集
class MultimodalDataset(Dataset):
    def __init__(self, file_path):
        # 使用json.load()解析整个json文件
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # 解析为Python对象

    def __len__(self):
        return len(self.data)  # 返回数据长度

    def __getitem__(self, idx):
        sample = self.data[idx]  # 获取数据

        # 根据数据是否存在来处理
        # text = sample.get('text')
        # image = sample.get('image_path')
        # data = sample.get('latest_value')
        #audio = sample.get('audio', None)
        #original = sample.get('original')

        return {
            'text': torch.tensor(sample['text'], dtype=torch.float32), #文本数据
            'image': torch.tensor(sample['image_path'], dtype=torch.float32), #图像数据
            'data': torch.tensor(sample['latest_value'], dtype=torch.float32), #时序数据
            #'audio': torch.tensor(audio, dtype=torch.float32) if audio is not None else None,
            #'original': original
        }

# 训练函数
def train_model(model, dataloader, optimizer, device):
    model.train()  # 训练模式
    total_loss = 0.0  # 初始化损失
    criterion = nn.MSELoss()  # 定义损失函数

    for batch in dataloader:
        text = batch['text'].to(device) if batch['text'] is not None else None
        image = batch['image'].to(device) if batch['image'] is not None else None
        data = batch['data'].to(device) if batch['data'] is not None else None
        #audio = batch['audio'].to(device) if batch['audio'] is not None else None
        original = batch['original'].to(device)  # 原始数据

        optimizer.zero_grad()  # 梯度清零

        # 前向传播，传入数据
        outputs = model(text, image, data) #audio

        # 计算MSE损失
        loss = criterion(outputs, original)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()  # 累加损失

    return total_loss / len(dataloader)  # 返回平均损失

# 验证函数
def validate_model(model, dataloader, criterion, device):
    model.eval()  # 评估模式
    total_loss = 0.0  # 初始化损失

    with torch.no_grad():  # 不计算梯度
        for batch in dataloader:
            text = batch['text'].to(device) if batch['text'] is not None else None
            image = batch['image'].to(device) if batch['image'] is not None else None
            data = batch['data'].to(device) if batch['data'] is not None else None
            #audio = batch['audio'].to(device) if batch['audio'] is not None else None
            original = batch['original'].to(device)  # 原始数据

            # 前向传播，传入数据
            outputs = model(text, image, data) #audio

            loss = criterion(outputs, original)  # 计算损失
            total_loss += loss.item()  # 累加损失

    return total_loss / len(dataloader)  # 返回平均损失

def main():
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_json = 'G://模型//data//news_data.json'  # 训练数据路径
    dataset = MultimodalDataset(train_json)  # 数据集

    # 动态划分训练集和验证集
    train_size = int(0.8 * len(dataset))  # 训练集大小为数据集的80%
    val_size = len(dataset) - train_size  # 验证集大小为数据集的20%
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  # 随机划分数据集

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练数据加载器
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # 验证数据加载器  

    # 加载模型
    model = MultimodalModel().to(device)  # 模型

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器, Adam优化器, 学习率为0.001
    
    # 训练设置
    num_epochs = 10  # 训练轮数
    checkpoint_path = 'G://模型//checkpoints'  # 模型保存路径
    os.makedirs(checkpoint_path, exist_ok=True)  # 创建模型保存路径

    # 训练
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss = validate_model(model, val_dataloader, optimizer, device)  # 需要传入损失函数

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # 打印训练信息，包括当前轮数，总轮数，训练损失，验证损失，保留4位小数

        # 保存模型
        checkpoint_path_epoch = os.path.join(checkpoint_path, f'model_{epoch + 1}.pth')
        # 模型保存路径，模型名称
        torch.save(model.state_dict(), checkpoint_path_epoch)

    print('Training Finished!')  # 训练结束

if __name__ == '__main__':
    main()  # 主函数
