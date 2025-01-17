import torch
import torch.nn as nn
import os
import torch.optim as optim #导入优化器
import argparse #导入参数解析器
from torch.utils.data import DataLoader,Dataset #导入数据加载器
from model.timesequence_model import TimeSequenceModel #导入时序模型


#数据集类
class PretrainTimeSequenceModel:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) #返回数据长度
    
    def __getitem__(self, idx):
        sample = self.data[idx] #获取数据

#预训练函数
def pretrain(model, train_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input in train_loader:
            #数据移动到设备
            input = input.to(device)
            #前向传播
            output = model(input)
            #定义损失函数
            loss = ((output - input) ** 2).mean() #均方误差
            #反向传播与优化
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播
            optimizer.step() #更新参数
            running_loss += loss.item() #每轮累加损失

        #输出epoch和损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

#参数解析函数
def parse_args():
     #创建参数解析器，设置参数解析
    parser = argparse.ArgumentParser(description="Pretrain TimeSequenceModel") #创建参数解析器
    parser.add_argument("--input_dim", type=int, default=768, help="输入维度") #输入维度
    parser.add_argument("--hidden_dim", type=int, default=32, help="隐藏层维度") #隐藏层维度
    parser.add_argument("--num_layers", type=int, default=2, help="LSTM层数") #LSTM层数
    parser.add_argument("--seq_len", type=int, default=5, help="序列长度") #序列长度
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout") #dropout 
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数") #训练轮数
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率") #学习率
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size") #batch_size
    parser.add_argument("--data_path", type=str, required=True, help="张量数据地址") #张量数据地址
    parser.add_argument("--save_path", type=str, default="time_sequence_model_pretrained.pth", help="模型保存地址") #模型保存地址
    return parser.parse_args() #返回参数

#主函数
def main():

    data_path = 'G://模型://data//news_data.json' #数据地址
    save_path = 'G://time_sequence_model_pretrained.pth' #模型

    #解析参数
    args = parse_args()

    #加载数据
    data_dict = torch.load(args.data_path) #加载数据
    data = data_dict['data'] #获取数据,假设键名为data

    #创建数据集
    dataset = PretrainTimeSequenceModel(data) #创建数据集
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) #创建数据加载器,shuffle=True意思是每次训练前打乱数据

    #初始化模型
    device = torch.devide("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSequenceModel(
        input_dim=args.input_dim, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers, 
        seq_len=args.seq_len, 
        dropout=args.dropout).to(device) #初始化模型
    
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #优化器, Adam优化器, 学习率为0.001

    #预训练模型
    pretrain(model, train_loader, optimizer, args.num_epochs, device)

    #保存模型
    torch.save(model.state_dict(), args.save_path) #保存模型
    print(f"Model saved at {args.save_path}") #打印模型保存地址

if __name__ == "__main__":
    main()