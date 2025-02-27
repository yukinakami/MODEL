import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MultimodalDataset
from model.model_Transformer import MultimodalModel
#from model.model_LSTM import MultimodalModel_LSTM
#from model.model_GRU import MultimodalModel_GRU
from sklearn.model_selection import train_test_split
import json
import logging
from tqdm import tqdm
import torch.nn.functional as F

# 配置日志
logging.basicConfig(filename='G://training_log_TRAN.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
def print_to_log(message):
    print(message)  # 继续打印到控制台
    logging.info(message)  # 同时将信息写入日志文件

# 数据集
dataset = MultimodalDataset("G://pretrain//selected_data.json")
#dataset = MultimodalDataset("G://模型//data//output.json")

# 切分数据集为训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# 初始化模型
model = MultimodalModel(text_model_name='bert-base-uncased', image_pretrained=True, hidden_dim=256)

# 选择优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 设备设置（如果有GPU可用，使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印模型信息
print_to_log(f"模型已预备并迁移至GPU||Model initialized and moved to {device}")

# 定义损失函数
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
huber_loss = nn.SmoothL1Loss()

# RMSE, R2, MAPE的计算函数
def rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))

def r2_score(output, target):
    total_variance = torch.var(target, unbiased=False)
    residual_variance = torch.var(target - output, unbiased=False)
    return 1 - residual_variance / total_variance

def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100

# 开始训练
num_epochs = 30
print_to_log("现在开始预训练||Starting training process...")

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    total_mape = 0
    num_batches = len(train_loader)

    for batch_idx, (text, image, data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # 将数据移动到 GPU/CPU
        text = {
            'input_ids': text['input_ids'].squeeze(1).to(device),
            'attention_mask': text['attention_mask'].squeeze(1).to(device)
        }
        image = image.to(device)
        data = data.to(device)
        target = target.to(device)

        # 目标张量形状调整
        target = target.squeeze(2)  # 形状调整为 [batch_size, 1]

        # 前向传播
        output = model(text, image, data)

        # 输出形状调整，去掉额外的维度 [batch_size, 1, 1] -> [batch_size, 1]
        output = output.squeeze(2)

        # 计算损失
        mse = mse_loss(output, target)
        mae = mae_loss(output, target)
        huber = huber_loss(output, target)
        rmse = rmse_loss(output, target)
        r2 = r2_score(output, target)
        mape = mape_loss(output, target)

        # 综合损失
        batch_loss = 0.5 * mse + 0.3 * mae + 0.2 * huber
        
        # 反向传播和优化
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累计评估指标
        total_loss += batch_loss.item()
        total_rmse += rmse.item()
        total_mae += mae.item()
        total_r2 += r2.item()
        total_mape += mape.item()

    # 输出训练集的平均损失
    avg_train_loss = total_loss / num_batches
    avg_train_rmse = total_rmse / num_batches
    avg_train_mae = total_mae / num_batches
    avg_train_r2 = total_r2 / num_batches
    avg_train_mape = total_mape / num_batches

    print_to_log(f"训练轮次||Epoch {epoch+1}/{num_epochs} - 损失||Training Loss: {avg_train_loss:.4f} - "
                  f"RMSE: {avg_train_rmse:.4f} - MAE: {avg_train_mae:.4f} - "
                  f"R²: {avg_train_r2:.4f} - MAPE: {avg_train_mape:.4f}")

    # 验证过程（每个 epoch 结束时）
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    val_rmse = 0
    val_mae = 0
    val_r2 = 0
    val_mape = 0
    num_val_batches = len(val_loader)

    with torch.no_grad():
        for text, image, data, target in val_loader:
            text = {
                'input_ids': text['input_ids'].squeeze(1).to(device),
                'attention_mask': text['attention_mask'].squeeze(1).to(device)
            }
            image = image.to(device)
            data = data.to(device)
            target = target.to(device)

            # 目标张量形状调整
            target = target.squeeze(2)  # 形状调整为 [batch_size, 1]

            # 前向传播
            output = model(text, image, data)

            # 输出形状调整
            output = output.squeeze(2)

            # 计算损失
            mse = mse_loss(output, target)
            mae = mae_loss(output, target)
            huber = huber_loss(output, target)
            rmse = rmse_loss(output, target)
            r2 = r2_score(output, target)
            mape = mape_loss(output, target)

            # 累计评估指标
            val_rmse += rmse.item()
            val_mae += mae.item()
            val_r2 += r2.item()
            val_mape += mape.item()

            # 综合损失
            val_loss += 0.5 * mse + 0.3 * mae + 0.2 * huber

    # 输出验证集的平均损失
    avg_val_loss = val_loss / num_val_batches
    avg_val_rmse = val_rmse / num_val_batches
    avg_val_mae = val_mae / num_val_batches
    avg_val_r2 = val_r2 / num_val_batches
    avg_val_mape = val_mape / num_val_batches

    print_to_log(f"验证轮次||Epoch {epoch+1}/{num_epochs} - 损失||Validation Loss: {avg_val_loss:.4f} - "
                  f"RMSE: {avg_val_rmse:.4f} - MAE: {avg_val_mae:.4f} - "
                  f"R²: {avg_val_r2:.4f} - MAPE: {avg_val_mape:.4f}")

# 保存模型
torch.save(model.state_dict(), "G://multimodal_model_TRAN.pth")
print_to_log("参数已保存至||Model saved to 'G://multimodal_model_TRAN.pth'")

print_to_log("训练结束||Training completed.")
