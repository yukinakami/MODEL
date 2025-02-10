import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MultimodalDataset  # 假设 Dataset 存储在 dataset.py 文件中
from model.model_m import MultimodalModel  # 假设模型存储在 model.py 文件中
from sklearn.model_selection import train_test_split
import json
import logging
from tqdm import tqdm  # 导入 tqdm 库

# 配置日志
logging.basicConfig(filename='G://training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
def print_to_log(message):
    print(message)  # 继续打印到控制台
    logging.info(message)  # 同时将信息写入日志文件

# 数据集
dataset = MultimodalDataset("G://pretrain//selected_data.json")
#dataset = MultimodalDataset("G://模型//data//output.json")

# 切分数据集为训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# 初始化模型
model = MultimodalModel(text_model_name='bert-base-uncased', image_pretrained=True, hidden_dim=256)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 设备设置（如果有GPU可用，使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印模型信息
print_to_log(f"模型已预备并迁移至GPU||Model initialized and moved to {device}")

# 定义损失函数
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.SmoothL1Loss()

# 开始训练
num_epochs = 20
print_to_log("现在开始预训练||Starting training process...")

for epoch in range(num_epochs):
    #print_to_log(f"Epoch {epoch+1}/{num_epochs}...")
    model.train()  # 设置模型为训练模式
    total_loss = 0

    # 使用 tqdm 包装 DataLoader，显示进度条
    for batch_idx, (text, image, data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # 将数据移动到 GPU/CPU
        text = {
            'input_ids': text['input_ids'].squeeze(1).to(device),
            'attention_mask': text['attention_mask'].squeeze(1).to(device)
        }
        image = image.to(device)
        data = data.to(device)
        target = target.to(device)

        # 前向传播
        output = model(text, image, data)

        # 计算每种损失
        mse = mse_loss(output, target)
        mae = mae_loss(output, target)
        huber = huber_loss(output, target)

        # 综合损失，可以调整每个损失的权重
        total_loss = 0.5 * mse + 0.3 * mae + 0.2 * huber
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清除之前的梯度
        total_loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        # 打印部分结果
        #print_to_log(f"Predictions: {output}")
        #print_to_log(f"Labels: {target}")
        #print(f"MSE Loss: {mse.item()}, MAE Loss: {mae.item()}, Huber Loss: {huber.item()}")

    print_to_log(f"训练轮次||Epoch {epoch+1}/{num_epochs} - 损失||Training Loss: {total_loss.item():.4f}")

    # 验证过程（每个 epoch 结束时）
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():  # 在验证时不需要计算梯度
        for text, image, data, target in val_loader:
            text = {
                'input_ids': text['input_ids'].squeeze(1).to(device),
                'attention_mask': text['attention_mask'].squeeze(1).to(device)
            }
            image = image.to(device)
            data = data.to(device)
            target = target.to(device)

            # 前向传播
            output = model(text, image, data)

            # 计算损失
            mse = mse_loss(output, target)
            mae = mae_loss(output, target)
            huber = huber_loss(output, target)

            # 综合损失
            val_loss += 0.5 * mse + 0.3 * mae + 0.2 * huber

    print_to_log(f"验证轮次||Epoch {epoch+1}/{num_epochs} - 损失||Validation Loss: {val_loss / len(val_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "G://multimodal_model.pth")
print_to_log("参数已保存至||Model saved to 'G://multimodal_model.pth'")

print_to_log("训练结束||Training completed.")
